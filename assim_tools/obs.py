import numpy as np

def prepare_obs(comm, obs_def, ):
    ##input: obs_def
    ##  obs[nobs],

    return obs, err, x_, y_, z_, t_, v_

# def prepare_obs_prior(obs, state):

#     return obs_prior

def prepare_impact(obs_def, state_def):
    ##set indices for state variables
    state_ind = {}
    for ind, state in enumerate(state_def):
        state_ind.update({state:ind})

    ##impact matrix size
    nvo = len(obs_def.keys())
    nv = len(state_def.keys())
    impact = np.zeros((nvo, nv), dtype=np.float32)
    ##parse obs_def and set entries impact matrix
    for ind, obs in enumerate(obs_def):
        obs_impact = obs_def[obs]['impact']
        for state in obs_impact:
            if state in state_def:
                impact[ind, state_ind[state]] = obs_impact[state]
    return impact
