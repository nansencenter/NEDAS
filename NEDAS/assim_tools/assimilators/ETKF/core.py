import numpy as np
from NEDAS.utils.njit import njit
from NEDAS.utils.parallel import bcast_by_root
from NEDAS.assim_tools.assimilators.batch import BatchAssimilator

class ETKFAssimilator(BatchAssimilator):
    random_rotation: bool
    transform_solver: str  # 'svd' or 'eigen'

    def assimilation_algorithm(self, c):
        # Generate ONE mean-preserving random orthogonal rotation per analysis
        # cycle, shared across all grid points and MPI ranks. DAPPER applies a
        # single global rotation (one analysis domain); applying independent
        # rotations at each grid point would scramble the spatial cross-
        # covariances of the analysis ensemble (G_x^T G_y != I) and cause filter
        # divergence. A single shared rotation relabels the members consistently
        # everywhere, preserving the full posterior covariance.
        if self.random_rotation and c.nens > 2:
            self.rotation_matrix = bcast_by_root(c.comm)(mean_preserving_rotation)(c.nens)
        else:
            self.rotation_matrix = np.eye(c.nens)
        super().assimilation_algorithm(c)

    def local_analysis(self, c, loc_id, ind, hlfactor, state_data, obs_data):
        state_var_id = state_data['var_id']  # variable id for each field (nfld)
        state_z = state_data['z'][:, loc_id]
        state_t = state_data['t'][:]

        # vertical, time and cross-variable (impact_on_variable) localization
        obs_value = obs_data['obs'][ind]
        obs_err = obs_data['err_std'][ind]
        obs_z = obs_data['z'][ind]
        obs_t = obs_data['t'][ind]
        obs_rec_id = obs_data['obs_rec_id'][ind]
        vroi = obs_data['vroi'][obs_rec_id]
        troi = obs_data['troi'][obs_rec_id]
        impact_on_variable = obs_data['impact_on_variable'][:, state_var_id][obs_rec_id]

        # the string solver option is mapped to a boolean here so that the njit
        # kernels do not need to perform string comparisons
        use_eigen = (self.transform_solver == 'eigen')

        local_analysis_main(state_data['state_prior'][...,loc_id], obs_data['obs_prior'][:,ind],
                            obs_value, obs_err, hlfactor,
                            state_z, obs_z, vroi, c.localization_funcs['vertical'],
                            state_t, obs_t, troi, c.localization_funcs['temporal'],
                            impact_on_variable, self.rotation_matrix, use_eigen)

@njit
def local_analysis_main(state_prior, obs_prior,
                        obs, obs_err, hlfactor,
                        state_z, obs_z, vroi, vlocal_func,
                        state_t, obs_t, troi, tlocal_func,
                        impact_on_variable, rotation, use_eigen) -> None:
    """perform local analysis for one location in the analysis grid partition"""
    nens, nfld = state_prior.shape
    nens_obs, nlobs = obs_prior.shape
    if nens_obs != nens:
        raise ValueError('Error: number of ensemble members in state and obs do not match!')

    lfactor_old = np.zeros(nlobs)
    weights_old = np.eye(nens)

    # loop through the field records
    for n in range(nfld):

        # vertical localization
        vdist = np.abs(obs_z - state_z[n])
        vlfactor = vlocal_func(vdist, vroi)
        if (vlfactor==0).all():
            continue  # the state is outside of vroi of all obs, skip

        # temporal localization
        tdist = np.abs(obs_t - state_t[n])
        tlfactor = tlocal_func(tdist, troi)
        if (tlfactor==0).all():
            continue  # the state is outside of troi of all obs, skip

        # total lfactor
        lfactor =  hlfactor * vlfactor * tlfactor * impact_on_variable[:, n]
        if (lfactor==0).all():
            continue

        # if prior spread is zero, don't update
        if np.std(state_prior[:, n]) == 0:
            continue

        # only need to assimilate obs with lfactor>0
        ind = np.where(lfactor>0)[0]

        # TODO:get rid of obs if obs_prior is nan
        # valid = np.array([np.isnan(obs_prior[:,i]).any() for i in ind])
        # ind = ind[valid]

        # sort the obs from high to low lfactor
        sort_ind = np.argsort(lfactor[ind])[::-1]
        ind = ind[sort_ind]

        # use cached weight if the localization factors are unchanged from the
        # previous field record, to avoid repeated computation. Note: when a
        # random rotation is applied, the cached weights (including their
        # rotation) are reused, keeping neighboring field records consistent.
        if n>0 and len(ind)==len(lfactor_old) and (lfactor[ind]==lfactor_old).all():
            weights = weights_old
        else:
            weights = ensemble_transform_weights(obs[ind], obs_err[ind], obs_prior[:, ind],
                                                 lfactor[ind], rotation, use_eigen)

        # perform local analysis and update the ensemble state
        state_prior[:, n] = apply_ensemble_transform(state_prior[:, n], weights)

        lfactor_old = lfactor[ind]
        weights_old = weights

@njit
def ensemble_transform_weights(obs, obs_err, obs_prior, local_factor,
                               rotation, use_eigen=False):
    """
    Compute the ETKF ensemble transform weight matrix for one local analysis.

    The algorithm follows the symmetric square-root (ETKF) formulation used in
    DAPPER (github.com/nansencenter/DAPPER, ``EnKF_analysis`` with the ``Sqrt``
    update). The transform is obtained directly from the singular value
    decomposition of the (whitened, localized) observation anomaly matrix S,
    rather than from forming the cross-product I + S^T S explicitly.

    ``rotation`` is a (nens x nens) mean-preserving orthogonal matrix applied to
    the symmetric square root (identity to disable). The SAME matrix must be
    used at every grid point of an analysis (see ``mean_preserving_rotation``).

    The returned ``weights`` matrix W (nens x nens) is such that the analysis
    ensemble is ``x_post[k] = sum_m x_prior[m] * W[m, k]``, with each column
    summing to one. W decomposes as ``W[m, k] = w[m] + T[m, k]`` where w is the
    mean-update weight and T is the (optionally rotated) symmetric square root.
    """
    nens, nlobs = obs_prior.shape

    # ensemble weight matrix, weights[:, m] is for the m-th member
    # also known as T in Bishop 2001, and X5 in Evensen textbook (and in Sakov 2012)
    weights = np.zeros((nens, nens))

    # find mean of obs_prior
    obs_prior_mean = np.zeros(nlobs)
    for m in range(nens):
        obs_prior_mean += obs_prior[m, :]
    obs_prior_mean /= nens

    obs_err_std = np.sqrt(obs_err**2)

    # whitened, localized observation anomaly matrix S (nens, nlobs) and
    # innovation dy (nlobs). Both are scaled by the localized R^{-1/2} and by
    # 1/sqrt(nens-1) so that the analysis Hessian in ensemble space is
    # (I + S S^T) with eigenvalues d = sv^2 + 1.
    S = np.zeros((nens, nlobs))
    dy = np.zeros(nlobs)
    for p in range(nlobs):
        S[:, p] = (obs_prior[:, p] - obs_prior_mean[p]) * local_factor[p] / obs_err_std[p]
        dy[p] = (obs[p] - obs_prior_mean[p]) * local_factor[p] / obs_err_std[p]
    S /= np.sqrt(nens-1)
    dy /= np.sqrt(nens-1)

    # TODO:factor in the correlated R in obs_err, need another SVD of R

    # ---decomposition of the analysis Hessian (I + S S^T) = U diag(d) U^T
    # The added I prevents rank issues when nlobs<nens: the eigenvalues that
    # would be zero in S S^T become 1, so the matrix is always invertible.
    if use_eigen:
        # eigen-decomposition of the explicitly formed nens x nens matrix
        hessian = np.eye(nens) + S @ S.T
        try:
            d, U = np.linalg.eigh(hessian)
        except Exception:
            # if the decomposition fails just return equal weights (no update)
            print('Error: failed to decompose the analysis Hessian')
            return np.eye(nens)
    else:
        # SVD performed on S itself (DAPPER style), more numerically stable than
        # forming the cross-product. Left singular vectors U (nens, nens) are the
        # eigenvectors of S S^T; the eigenvalues of the Hessian are sv^2 + 1.
        try:
            U, sv, _ = np.linalg.svd(S, full_matrices=True)
        except Exception:
            # if the decomposition fails just return equal weights (no update)
            print('Error: failed to compute SVD of S')
            return np.eye(nens)
        d = np.ones(nens)
        for i in range(sv.size):
            d[i] += sv[i]**2

    d_inv = 1.0 / d
    d_inv_sqrt = np.sqrt(d_inv)

    # var_ratio = (I + S S^T)^{-1} = U diag(d^{-1}) U^T
    var_ratio = (U * d_inv) @ U.T

    # ----first part of weights: update of the ensemble mean
    # w = var_ratio @ S @ dy, the Kalman-gain weight applied to the innovation
    w = var_ratio @ (S @ dy)

    # ----second part of weights: symmetric square root T = U diag(d^{-1/2}) U^T
    var_ratio_sqrt = (U * d_inv_sqrt) @ U.T

    # ----mean-preserving random orthogonal rotation (DAPPER genOG_1)
    # DAPPER post-multiplies the symmetric square root as T <- G @ T. Here the
    # weight matrix is the transpose of DAPPER's transform (NEDAS applies
    # x_post[k] = sum_m x_prior[m] W[m,k]), so the rotation is applied on the
    # right as T <- T @ G^T. Because G fixes the ones-vector, this preserves
    # both the analysis covariance (T^T T) and the column sums of W. When
    # rotation is the identity matrix this is a no-op.
    var_ratio_sqrt = var_ratio_sqrt @ rotation.T

    # assemble W[m, k] = w[m] + T[m, k]
    weights += var_ratio_sqrt
    for m in range(nens):
        weights[m, :] += w[m]

    return weights

@njit
def random_orthogonal_matrix(m):
    """Generate a random orthogonal matrix in O(m) via QR of a Gaussian matrix.

    Equivalent to DAPPER's genOG: the columns are normalized so that the
    diagonal of R is positive, giving a uniform (Haar) distribution.
    """
    H = np.random.standard_normal((m, m))
    Q, R = np.linalg.qr(H)
    for i in range(m):
        if R[i, i] < 0:
            Q[:, i] = -Q[:, i]
    return Q

@njit
def mean_preserving_rotation(nens):
    """Random orthogonal matrix that fixes the ones-vector (DAPPER's genOG_1).

    Constructs ``V @ block_diag(1, Q) @ V.T`` where V is an orthonormal basis
    whose first column is proportional to ones, and Q is a random orthogonal
    matrix in O(nens-1). The result U satisfies ``U @ ones == ones``, so applying
    it to the ensemble anomalies leaves the ensemble mean unchanged.
    """
    # orthonormal basis whose first column is proportional to ones
    e = np.ones((nens, 1))
    V, _, _ = np.linalg.svd(e, full_matrices=True)

    # block_diag(1, Q): keep the ones-direction fixed, rotate the complement
    block = np.eye(nens)
    Q = random_orthogonal_matrix(nens - 1)
    block[1:, 1:] = Q

    return V @ block @ V.T

@njit
def apply_ensemble_transform(ens_prior, weights):
    """Apply the weights to transform local ensemble"""

    nens = ens_prior.size
    ens_post = ens_prior.copy()

    # check if weights sum to 1
    for m in range(nens):
        sum_wgts = np.sum(weights[:, m])
        if np.abs(sum_wgts - 1) > 1e-5:
            raise RuntimeError('ETKF: sum of weights != 1 detected! Aborting...')

    # apply the weights
    for m in range(nens):
        ens_post[m] = np.sum(ens_prior * weights[:, m])

    return ens_post
