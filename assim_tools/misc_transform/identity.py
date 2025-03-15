class Transform:
    """
    Base class for misc. transforms: no transform
    """
    def forward_state(self, c, rec, field):
        return field

    def backward_state(self, c, rec, field):
        return field

    def forward_obs(self, c, obs_rec, obs_seq):
        return obs_seq

    def backward_obs(self, c, obs_rec, obs_seq):
        return obs_seq

