import numpy as np
from utils.njit import njit
from .batch import BatchAssimilator

class TopazDEnKFAssimilator(BatchAssimilator):

    @classmethod
    def ensemble_transform_weights(cls, *args, **kwargs):
        return cls._ensemble_transform_weights(*args, **kwargs)
    
    @staticmethod
    @njit(cache=True)
    def _ensemble_transform_weights(obs, obs_err, obs_prior, filter_type, local_factor, rfactor, kfactor):
        """
        Compute the transform weights for the local ensemble

        Inputs:
        - obs: np.array[nlobs]
        The local observation sequence

        - obs_err: np.array[nlobs]
        The observation error, sqrt(R) vector

        - obs_prior: np.array[nens, nlobs]
        The observation priors

        - filter_type: str
        Type of filter to use: "ETKF", or "DEnKF"

        - local_factor: np.array[nlobs]
        Localization/impact factor for each observation
        
        - rfactor: float

        - kfactor: float

        Return:
        - weights: np.array[nens, nens]
        The ensemble transform weights
        """
        nens, nlobs = obs_prior.shape

        ##ensemble weight matrix, weights[:, m] is for the m-th member
        ##also known as T in Bishop 2001, and X5 in Evensen textbook (and in Sakov 2012)
        weights = np.zeros((nens, nens))

        ##find mean of obs_prior
        obs_prior_mean = np.zeros(nlobs)
        for m in range(nens):
            obs_prior_mean += obs_prior[m, :]
        obs_prior_mean /= nens
        ##find variance of obs_prior
        obs_prior_var = np.zeros(nlobs)
        for m in range(nens):
            obs_prior_var += (obs_prior[m, :] - obs_prior_mean)**2
        obs_prior_var /= nens-1
        
        innov = obs - obs_prior_mean
        obs_var = obs_err**2

        ##inflate obs error by rfactor, and kfactor where innovation is large
        obs_var = np.sqrt((obs_prior_var + obs_var)**2 + obs_prior_var*(innov/kfactor)**2) - obs_prior_var
        obs_err = np.sqrt(obs_var)

        ##obs_prior_pert S and innovation dy, normalized by sqrt(nens-1) R^-0.2
        S = np.zeros((nlobs, nens))
        dy = np.zeros((nlobs))
        for p in range(nlobs):
            S[p, :] = (obs_prior[:, p] - obs_prior_mean[p]) * local_factor[p] / obs_err[p]
            dy[p] = (obs[p] - obs_prior_mean[p]) * local_factor[p] / obs_err[p]
        S /= np.sqrt(nens-1)
        dy /= np.sqrt(nens-1)

        ##TODO:factor in the correlated R in obs_err, need another SVD of R
        # obs_err_corr

        ##----first part of weights: update of mean
        ##find singular values of the inverse variance ratio matrix (I + S^T S)
        ##note: the added I actually helps prevent issues if S^T S is not full rank
        ##      when nlobs<nens, there will be singular values of 0, but the full matrix
        ##      can still be inverted with singular values of 1.
        var_ratio_inv = np.eye(nens) + S.T @ S

        ##TODO: var_ratio_inv = np.eye(nlobs) + S @ S.T  if nlobs<nens

        try:
            L, sv, Rh = np.linalg.svd(var_ratio_inv)
        except:
            ##if svd failed just return equal weights (no update)
            print('Error: failed to invert var_ratio_inv=', var_ratio_inv)
            return np.eye(nens)

        ##the update of ens mean is given by (I + S^T S)^-1 S^T dy
        ##namely, var_ratio * obs_prior_var / obs_var * dy = G dy
        var_ratio = L @ np.diag(sv**-1) @ Rh
        
        ##the gain matrix
        gain = var_ratio @ S.T

        for m in range(nens):
            weights[m, :] = np.sum(gain[m, :] * dy)

        ##---second part of weights: update of ensemble spread
        if rfactor > 1:  ##scale S with rfactor and recompute the var_ratio
            S /= np.sqrt(rfactor)
            try:
                var_ratio_inv = np.eye(nens) + S.T @ S
                L, sv, Rh = np.linalg.svd(var_ratio_inv)
                var_ratio = L @ np.diag(sv**-1) @ Rh
                gain = var_ratio @ S.T
            except:
                ##if failed just return equal weights (no update)
                return np.eye(nens)

        if filter_type == 'ETKF':
            ##the update of ens pert is (I + S^T S)^-0.5, namely sqrt(var_ratio)
            var_ratio_sqrt = L @ np.diag(sv**-0.5) @ Rh

        elif filter_type == 'DEnKF':
            ##take Taylor approx. of var_ratio_sqrt (Sakov 2008)
            var_ratio_sqrt = np.eye(nens) - 0.5 * gain @ S

        else:
            print('Error: unknown filter type: '+filter_type)
            raise ValueError

        weights += var_ratio_sqrt

        return weights

    @classmethod
    def apply_ensemble_transform(cls, *args, **kwargs):
        return cls._apply_ensemble_transform(*args, **kwargs)

    @staticmethod
    @njit(cache=True)
    def _apply_ensemble_transform(ens_prior, weights):
        """Apply the weights to transform local ensemble"""

        nens = ens_prior.size
        ens_post = ens_prior.copy()

        ##check if weights sum to 1
        for m in range(nens):
            sum_wgts = np.sum(weights[:, m])
            if np.abs(sum_wgts - 1) > 1e-5:
                print('Warning: sum of weights != 1 detected!')

        ##apply the weights
        for m in range(nens):
            ens_post[m] = np.sum(ens_prior * weights[:, m])

        return ens_post

