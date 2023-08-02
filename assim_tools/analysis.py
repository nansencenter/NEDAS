import numpy as np
from numba import njit

###DA local analysis step
##inputs: state_info: dict[varname, variable names along the nfield dimension
##                            time, t2h(datetime obj)
##                          levels, z index of vertical levels
##                      local_inds, index of horizontal grid points]
##        state_ens_prior: [nens, nfield, len(local_inds)]
##        obs_seq[nlobs]: obs_rec[varname, observation variable name along nlobs
def local_analysis(state_info, state_ens_prior, obs_seq, filter_type='', ):

    state_ens_post = state_ens_prior.copy()


###two-step solution, Anderson 2003, used in DART
###obs increment
# def obs_incr

###updating state variable using obs_incr
# def reg_factor
    return state_ens_post
