
# def get_state(c):
#     return State(c)

# def get_obs(c):
#     return Obs(c)

# def get_covariance(c):
#     return Covariance(c)

def get_assimilator(c):
    if c.assim_mode == 'batch':
        if c.filter_type == 'TopazDEnKF':
            from .assimilators.TopazDEnKF import TopazDEnKFAssimilator as Assimilator
        elif c.filter_type == 'ETKF':
            from .assimilators.ETKF import ETKFAssimilator as Assimilator
        else:
            raise ValueError(f"Unknown filter_type {c.filter_type} for batch assimilation")
    elif c.assim_mode == 'serial':
        if c.filter_type == 'EAKF':
            from .assimilators.EAKF import EAKFAssimilator as Assimilator
#        elif c.filter_type == 'RHF':
#            from .assimilators.RHF import RHFAssimilator as Assimilator
#        elif c.filter_type == 'QCEKF':
#            from .assimilators.QCEKF import QCEKFAssimilator as Assimilator
        else:
            raise ValueError(f"Unknown filter_type {c.filter_type} for serial assimilation")
    else:
        raise ValueError(f"Unknown assim_mode {c.assim_mode}")
    return Assimilator()

def get_updator(c):
    if c.run_alignment:
        from .updators.alignment import AlignmentUpdator as Updator
    else:
        from .updators.additive import AdditiveUpdator as Updator
    return Updator()

# def get_localization(c):
#     return Localization(c)

# def get_inflation(c):
#     return Inflation(c)
