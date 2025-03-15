class Transform:
    """
    Base class for misc. transforms: no transform
    """
    def forward_state(self, c, rec, field):
        pass

    def backward_state(self, c, rec, field):
        pass

    def forward_obs(self, c, obs_rec, obs_seq):
        pass

    def backward_obs(self, c, obs_rec, obs_seq):
        pass

