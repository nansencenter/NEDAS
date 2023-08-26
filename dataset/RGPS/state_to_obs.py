def seaice_drift(path, obs_seq, **kwargs):
    return 1


state_to_obs = {'seaice_drift':seaice_drift,
               }
