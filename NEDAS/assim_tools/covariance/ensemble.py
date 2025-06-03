import yaml
import numpy as np
from NEDAS.utils.njit import njit

class EnsembleCovariance:
    def compute_covariance(self,):
        pass

# def covariance(x, y, cov_model='ensemble'):
#     if cov_model == 'ensemble':
#         return None
#     elif cov_model == 'static':
#         return None

@njit
def ensemble_covariance(state_ens, obs_ens):
    """
    Compute ensemble-based covariance between state and observation variables
    Inputs:
    - state_ens: np.ndarray
    State variables, first dimension is ensemble members
    - obs_ens: np.ndarray
    Observation variables, first dimension is ensemble members
    Return:
    - state_variance: np.ndarray, shape is same as state_ens.shape[1:]
    Variance of state variables
    - obs_variance: np.ndarray, shape is same as obs_ens.shape[1:]
    Variance of observation variables
    - corr: np.ndarray, shape is state_ens.shape[1:]+obs_ens.shape[1:]
    Correlation between state and observation variables
    """
    nens = state_ens.shape[0]
    nens_obs = obs_ens.shape[0]
    if nens != nens_obs:
        raise ValueError('Ensemble sizes of state and observation must be equal')
    
    # compute variance of state variables
    state_variance = np.zeros(state_ens.shape[1:])
    for m in range(nens):
        state_variance += state_ens[m,...]**2
    state_variance /= nens-1
    
    # compute variance of observation variables
    obs_variance = np.zeros(obs_ens.shape[1:])
    for m in range(nens):
        obs_variance += obs_ens[m,...]**2
    obs_variance /= nens-1

    # compute covariance between state and observation variables
    new_shape = state_ens.shape[1:]+obs_ens.shape[1:]
    cov = np.zeros(new_shape)
    for m in range(nens):
        cov += np.outer(state_ens[m,...], obs_ens[m,...]).reshape(new_shape)
    cov /= nens-1

    # correlation between state and observation variables
    corr = cov / np.sqrt(np.outer(state_variance, obs_variance).reshape(new_shape))
    
    return state_variance, obs_variance, corr

#def static_covariance(lookup, h_dist, v_dist, t_dist, ):
#
#    return state

def covariance_lookup(c):
    pass
