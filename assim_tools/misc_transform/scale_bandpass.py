from grid import Grid
from utils.multiscale import get_scale_component, get_error_scale_factor

class ScaleBandpassTransform:
    """
    Base class for misc. transforms: no transform
    """
    def forward_state(self, c, rec, field):
        field = get_scale_component(c.grid, field, c.character_length, c.scale_id)
        return field

    def backward_state(self, c, rec, field):
        return field

    def forward_obs(self, c, obs_rec, obs_seq):
        obs_grid = Grid(c.grid.proj, obs_seq['x'], obs_seq['y'], regular=False)
        obs_seq['obs'] = get_scale_component(obs_grid, obs_seq['obs'], c.character_length, c.scale_id)
        obs_seq['err_std'] *= get_error_scale_factor(obs_grid, c.character_length, c.scale_id)
        return obs_seq

    def backward_obs(self, c, obs_rec, obs_seq):
        return obs_seq
