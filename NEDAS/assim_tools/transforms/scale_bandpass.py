import numpy as np
from NEDAS.grid import Grid
from NEDAS.utils.multiscale import get_scale_component, get_error_scale_factor
from .base import Transform

class ScaleBandpass(Transform):
    """
    Subclass for scale bandpass filter to get a scale component.
    """
    def __init__(self, decompose_obs=True, **kwargs):
        self.decompose_obs = decompose_obs

    def forward_state(self, c, rec, field):
        if c.nscale == 1:
            return field

        ##for state on analysis grid, just get_scale_component (using Fourier method)
        ##pad voids with zero
        mask = np.isnan(field)
        field[mask] = 0.0
        field = get_scale_component(c.grid, field, c.character_length, c.iter)
        field[mask] = np.nan
        return field

    def backward_state(self, c, rec, field):
        return field

    def forward_obs(self, c, obs_rec, obs_seq):
        if c.nscale == 1:
            return obs_seq

        if not self.decompose_obs:
            return obs_seq

        ##temporarily convert obs grid to the analysis grid
        ##create the irregular obs grid
        obs_grid = Grid(c.grid.proj, obs_seq['x'], obs_seq['y'], regular=False)

        ##remove unwanted triangles in the obs grid
        max_a = np.quantile(obs_grid.tri.a, 0.999)
        max_p = np.quantile(obs_grid.tri.p, 0.99)
        msk = np.logical_or(obs_grid.tri.a > max_a, obs_grid.tri.p > max_p, obs_grid.tri.ratio < 0.3)

        ##convert obs to analysis grid
        obs_grid = Grid(c.grid.proj, obs_seq['x'], obs_seq['y'], regular=False, triangles=obs_grid.tri.triangles[~msk,:])
        obs_grid.set_destination_grid(c.grid)
        obs_fld = obs_grid.convert(obs_seq['obs'], is_vector=obs_rec['is_vector'], method='nearest', coarse_grain=False)

        ##pad voids with zeros, for convolution later
        mask = np.isnan(obs_fld)
        obs_fld[mask] = 0.0

        ##get scale component on analysis grid
        obs_fld_new = get_scale_component(c.grid, obs_fld, c.character_length, c.iter)
        if obs_rec['is_vector']:
            for i in range(2):
                obs_seq['obs'][i,...] = c.grid.interp(obs_fld_new[i,...], obs_seq['x'], obs_seq['y'], method='nearest')
        else:
            obs_seq['obs'] = c.grid.interp(obs_fld_new, obs_seq['x'], obs_seq['y'], method='nearest')

        ##TODO: current implementation is very slow
        ##update obs err std because some averaging happened in get_scale_component
        #obs_seq['err_std'] *= get_error_scale_factor(c.grid, c.character_length, c.iter)
        obs_seq['err_std'] *= c.obs_err_scale_fac[c.iter]

        return obs_seq

    def backward_obs(self, c, obs_rec, obs_seq):
        return obs_seq
