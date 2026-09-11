import numpy as np
from NEDAS.utils.njit import njit
from NEDAS.utils.parallel import bcast_by_root
from NEDAS.assim_tools.assimilators.batch import BatchAssimilator

class ETKFAssimilator(BatchAssimilator):
    random_rotation: bool
    transform_solver: str  # 'svd', 'eigen', or 'auto'
    supports_hybrid_perturbation = True

    def assimilation_algorithm(self, c):
        # scaling of the dynamic/static anomalies that gives the hybrid covariance
        self.anomaly_factors = c.covariance.anomaly_factors()

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
        self._warned_eigen_fallback = False
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
        # kernels do not need to perform string comparisons. 'auto' (default,
        # unset by the user) picks eigen when nlobs >> nens, since svd's
        # full_matrices=True is O(nlobs^3) there vs eigen's O(nens^3) -- an
        # explicit 'svd' or 'eigen' choice is always respected as-is.
        if self.transform_solver == 'auto':
            use_eigen = len(ind) > 4 * (c.nens + c.nens_static)
            if use_eigen and not self._warned_eigen_fallback:
                c.debug_message = 'ETKF: nlobs >> nens, auto-selected eigen transform_solver'
                self._warned_eigen_fallback = True
        else:
            use_eigen = (self.transform_solver == 'eigen')

        local_analysis_main(state_data['state_prior'][...,loc_id], obs_data['obs_prior'][:,ind],
                            state_data['state_static'][...,loc_id], obs_data['obs_prior_static'][:,ind],
                            obs_value, obs_err, hlfactor,
                            state_z, obs_z, vroi, c.localization_funcs['vertical'],
                            state_t, obs_t, troi, c.localization_funcs['temporal'],
                            impact_on_variable, self.rotation_matrix, use_eigen,
                            *self.anomaly_factors, c.covariance.hybrid_perturbation)

@njit
def local_analysis_main(state_prior, obs_prior, state_static, obs_prior_static,
                        obs, obs_err, hlfactor,
                        state_z, obs_z, vroi, vlocal_func,
                        state_t, obs_t, troi, tlocal_func,
                        impact_on_variable, rotation, use_eigen,
                        fac_dynamic, fac_static, hybrid_perturbation) -> None:
    """
    perform local analysis for one location in the analysis grid partition, updating the
    dynamic members in state_prior; the static members (state_static, obs_prior_static)
    enter through the hybrid covariance, see ensemble_transform_weights
    """
    nens, nfld = state_prior.shape
    nens_obs, nlobs = obs_prior.shape
    if nens_obs != nens:
        raise ValueError('Error: number of ensemble members in state and obs do not match!')
    nens_static = state_static.shape[0]
    if obs_prior_static.shape[0] != nens_static:
        raise ValueError('Error: number of static members in state and obs do not match!')

    lfactor_old = np.zeros(nlobs)
    weights_old = np.eye(nens)
    weights_static_old = np.zeros((nens_static, nens))

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

        # if prior spread is zero (in both the dynamic and the static members), don't update
        if np.std(state_prior[:, n]) == 0 and (nens_static == 0 or np.std(state_static[:, n]) == 0):
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
            weights_static = weights_static_old
        else:
            weights, weights_static = ensemble_transform_weights(obs[ind], obs_err[ind],
                                                                 obs_prior[:, ind], obs_prior_static[:, ind],
                                                                 lfactor[ind], rotation, use_eigen,
                                                                 fac_dynamic, fac_static, hybrid_perturbation)

        # perform local analysis and update the ensemble state,
        # the static members contribute to the dynamic members' update through weights_static
        state_prior[:, n] = apply_ensemble_transform(state_prior[:, n], weights)
        for m in range(nens_static):
            state_prior[:, n] += state_static[m, n] * weights_static[m, :]

        lfactor_old = lfactor[ind]
        weights_old = weights
        weights_static_old = weights_static

@njit
def ensemble_transform_weights(obs, obs_err, obs_prior, obs_prior_static, local_factor,
                               rotation, use_eigen, fac_dynamic, fac_static, hybrid_perturbation):
    """
    Compute the ETKF ensemble transform weights for one local analysis.

    The algorithm follows the symmetric square-root (ETKF) formulation used in
    DAPPER (github.com/nansencenter/DAPPER, ``EnKF_analysis`` with the ``Sqrt``
    update), working with the decomposition of the (whitened, localized)
    observation anomaly matrix S (see ``hessian_decomposition``).

    Hybrid covariance (see assim_tools/covariance): obs_prior (nens, nlobs) is from the
    dynamic members and obs_prior_static (nens_static, nlobs) from the static members.
    Their anomalies A_d, A_s (each about its own mean) scaled by ``fac_dynamic`` and
    ``fac_static`` form Z = [fac_dynamic*A_d, fac_static*A_s], with
    Z Z^T = P = (1-beta)*P_d + beta*alpha*P_s. The ensemble mean is updated with P
    (hybrid ETKF-OI, Wang et al. 2007); the dynamic perturbations are updated
      - hybrid_perturbation=False: by the ETKF transform of the dynamic ensemble alone,
        the static covariance only affects the mean (Wang et al. 2007);
      - hybrid_perturbation=True: by the reduced Kalman gain of P (Whitaker and Hamill 2002),
        A_d <- A_d - K~ H A_d, the deterministic hybrid EnKF-OI of Counillon et al. 2009.
    The plain ETKF is no static members with fac_dynamic = 1/sqrt(nens-1); with beta=0
    both options reduce to it.

    ``rotation`` is a (nens x nens) mean-preserving orthogonal matrix applied to
    the square root (identity to disable). The SAME matrix must be used at every
    grid point of an analysis (see ``mean_preserving_rotation``).

    Returns ``weights`` (nens, nens) and ``weights_static`` (nens_static, nens), such that the
    analysis ensemble is
    ``x_post[k] = sum_m x_prior[m] * weights[m, k] + sum_j x_static[j] * weights_static[j, k]``,
    the columns of weights sum to one and those of weights_static to zero. They decompose as
    ``weights[m, k] = w[m] + pert_weights[m, k]``, w the mean-update weight and pert_weights the
    (optionally rotated) square root for the perturbations, and likewise for weights_static.
    """
    nens, nlobs = obs_prior.shape
    nens_static = obs_prior_static.shape[0]

    # obs prior mean of the dynamic and the static members
    obs_prior_mean = np.zeros(nlobs)
    for m in range(nens):
        obs_prior_mean += obs_prior[m, :]
    obs_prior_mean /= nens
    obs_prior_mean_static = np.zeros(nlobs)
    for m in range(nens_static):
        obs_prior_mean_static += obs_prior_static[m, :]
    if nens_static > 0:
        obs_prior_mean_static /= nens_static

    obs_err_std = np.sqrt(obs_err**2)

    # whitened, localized obs anomalies of the dynamic (nens, nlobs) and the static
    # (nens_static, nlobs) members, each about its own mean, and innovation dy (nlobs) w.r.t.
    # the dynamic mean. S stacks the dynamic rows and then the static rows, scaled by their
    # anomaly factors: it is the whitened H Z, so the analysis Hessian in ensemble space is (I + S S^T).
    obs_anomaly = np.zeros((nens, nlobs))
    obs_anomaly_static = np.zeros((nens_static, nlobs))
    dy = np.zeros(nlobs)
    for p in range(nlobs):
        whitening_factor = local_factor[p] / obs_err_std[p]
        obs_anomaly[:, p] = (obs_prior[:, p] - obs_prior_mean[p]) * whitening_factor
        obs_anomaly_static[:, p] = (obs_prior_static[:, p] - obs_prior_mean_static[p]) * whitening_factor
        dy[p] = (obs[p] - obs_prior_mean[p]) * whitening_factor
    S = np.zeros((nens + nens_static, nlobs))
    S[:nens, :] = obs_anomaly * fac_dynamic
    S[nens:, :] = obs_anomaly_static * fac_static

    # TODO:factor in the correlated R in obs_err, need another SVD of R

    success, d, U = hessian_decomposition(S, use_eigen)
    if not success:
        # if the decomposition fails just return equal weights (no update)
        return np.eye(nens), np.zeros((nens_static, nens))

    d_inv = 1.0 / d
    d_inv_sqrt = np.sqrt(d_inv)

    # var_ratio = (I + S S^T)^{-1} = U diag(d^{-1}) U^T
    var_ratio = (U * d_inv) @ U.T

    # ----first part of weights: update of the ensemble mean with P
    # mean increment = Z var_ratio S dy, the Kalman-gain weight applied to the innovation;
    # as member weights, w (dynamic) and w_static (static) each sum to zero
    mean_gain_weights = var_ratio @ (S @ dy)
    w = fac_dynamic * mean_gain_weights[:nens]
    w_static = fac_static * mean_gain_weights[nens:]

    # ----second part of weights: square root for the dynamic perturbations, the columns
    # of pert_weights sum to one (carrying the dynamic mean), those of pert_weights_static to zero
    pert_weights_static = np.zeros((nens_static, nens))
    if hybrid_perturbation:
        # K~ H A_d in ensemble space is Z reduced_gain_weights, with
        # reduced_gain_weights = f(S S^T) S obs_anomaly^T and f(d) = 1/(sqrt(d) (1+sqrt(d))),
        # the Whitaker and Hamill (2002) reduced gain pushed through to ensemble space.
        # At beta=0, I - fac_dynamic * reduced_gain_weights = (I + S S^T)^{-1/2}.
        reduced_gain_factor = d_inv_sqrt / (1.0 + np.sqrt(d))
        reduced_gain_weights = (U * reduced_gain_factor) @ U.T @ (S @ obs_anomaly.T)
        pert_weights = np.eye(nens) - fac_dynamic * reduced_gain_weights[:nens, :]
        pert_weights_static = -fac_static * reduced_gain_weights[nens:, :]
    else:
        # symmetric square root of the dynamic ensemble alone, U diag(d^{-1/2}) U^T;
        # the decomposition above is reused when S is the dynamic ensemble alone (plain ETKF)
        if nens_static > 0 or fac_dynamic != 1.0 / np.sqrt(max(nens - 1, 1)):
            success, d, U = hessian_decomposition(obs_anomaly / np.sqrt(max(nens - 1, 1)), use_eigen)
            if not success:
                return np.eye(nens), np.zeros((nens_static, nens))
            d_inv_sqrt = np.sqrt(1.0 / d)
        var_ratio_sqrt = (U * d_inv_sqrt) @ U.T
        pert_weights = var_ratio_sqrt

    # ----mean-preserving random orthogonal rotation (DAPPER genOG_1)
    # DAPPER post-multiplies the symmetric square root as T <- G @ T. Here the
    # weight matrix is the transpose of DAPPER's transform (NEDAS applies
    # x_post[k] = sum_m x_prior[m] W[m,k]), so the rotation is applied on the
    # right as T <- T @ G^T. Because G fixes the ones-vector, this preserves
    # both the analysis covariance (T^T T) and the column sums of W. When
    # rotation is the identity matrix this is a no-op.
    pert_weights = pert_weights @ rotation.T
    if nens_static > 0:
        pert_weights_static = pert_weights_static @ rotation.T

    # ensemble weight matrix, weights[:, k] is for the k-th member
    # also known as T in Bishop 2001, and X5 in Evensen textbook (and in Sakov 2012)
    weights = np.zeros((nens, nens))
    weights_static = np.zeros((nens_static, nens))
    for k in range(nens):
        weights[:, k] = w + pert_weights[:, k]
        weights_static[:, k] = w_static + pert_weights_static[:, k]

    return weights, weights_static

@njit
def hessian_decomposition(S, use_eigen):
    """
    Decomposition of the analysis Hessian in ensemble space (I + S S^T) = U diag(d) U^T,
    for the whitened obs anomaly matrix S (nens, nlobs).

    The added I prevents rank issues when nlobs<nens: the eigenvalues that would be
    zero in S S^T become 1, so the matrix is always invertible.

    Returns (success, d, U), success is False if the decomposition failed.
    """
    nens = S.shape[0]
    if use_eigen:
        # eigen-decomposition of the explicitly formed nens x nens matrix
        try:
            d, U = np.linalg.eigh(np.eye(nens) + S @ S.T)
        except Exception:
            print('Error: failed to decompose the analysis Hessian')
            return False, np.ones(nens), np.eye(nens)
    else:
        # SVD performed on S itself (DAPPER style), more numerically stable than
        # forming the cross-product. Left singular vectors U (nens, nens) are the
        # eigenvectors of S S^T; the eigenvalues of the Hessian are sv^2 + 1.
        try:
            U, sv, _ = np.linalg.svd(S, full_matrices=True)
        except Exception:
            print('Error: failed to compute SVD of S')
            return False, np.ones(nens), np.eye(nens)
        d = np.ones(nens)
        for i in range(sv.size):
            d[i] += sv[i]**2
    return True, d, np.ascontiguousarray(U)

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

    # Renormalize each column to sum to exactly 1. The SVD/eigen-based construction
    # of `weights` is only mean-preserving up to floating-point precision -- in
    # practice this drift is small (~1e-5) and uniform across members, not a sign of
    # an ill-conditioned or wrongly-computed transform (the underlying d/U from
    # ensemble_transform_weights remain well-conditioned), but it occurs on nearly
    # every local analysis, so aborting on it (as before) made ETKF unusable for
    # anything but toy problems. Correct it explicitly rather than raising.
    for m in range(nens):
        weights[:, m] /= np.sum(weights[:, m])

    # apply the weights
    for m in range(nens):
        ens_post[m] = np.sum(ens_prior * weights[:, m])

    return ens_post
