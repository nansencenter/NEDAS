import numpy as np
from NEDAS.grid import Grid, IrregularGrid, Grid1D
from NEDAS.utils.multiscale import get_scale_component, get_error_scale_factor
from NEDAS.core import Context, Transform

class ScaleBandpass(Transform):
    """
    Subclass for scale bandpass filter to get a scale component.
    """
    nscale: int
    decompose_obs: bool
    character_length: list[float]

    def __init__(self, c, decompose_obs=True, character_length=None, **kwargs):
        self.decompose_obs = decompose_obs

        # validate config parameters
        self.nscale = c.config.niter
        # forward_state/forward_obs both no-op when nscale==1 (see below) -- see Transform.is_identity
        self.is_identity = (self.nscale == 1)

        # character_length is this transform's own parameter (given directly in its
        # transform_def entry), not a standalone top-level config field -- it belongs here
        # because it only means something in the context of a real spectral decomposition;
        # resolution_level stays a top-level config field since core/context.py uses it
        # generically every iteration regardless of which transform is active.
        assert isinstance(character_length, list), f"{character_length} is not a list"
        assert len(character_length) == self.nscale, f"{character_length} length != {self.nscale}"
        self.character_length = character_length

        value = c.config.resolution_level
        assert isinstance(value, list), f"{value} is not a list"
        assert len(value) == self.nscale, f"{value} length != {self.nscale}"

    def forward_state(self, c, rec, field):
        if self.nscale == 1:
            return field

        # for state on analysis grid, just get_scale_component (using Fourier method)
        # pad voids with zero
        mask = np.isnan(field)
        field[mask] = 0.0
        field = get_scale_component(c.grid, field, self.character_length, c.iter)
        field[mask] = np.nan
        return field

    def backward_state(self, c, rec, field):
        return field

    def forward_obs(self, c, obs_rec, obs_seq):
        if self.nscale == 1:
            return obs_seq

        # per-scale obs err std (e.g. Ying 2019's narrower-scale err inflation, to account for
        # unfiltered obs innovations partly reflecting content from other scales) is now the
        # obs_def's own responsibility: obs_rec.err.std can be given as a per-iteration
        # 'iter0'/'iter1'/... dict (resolved once, fresh, in ObsInfo.add_obs_record each
        # iteration) -- already baked into obs_seq['err_std'] by collect_obs_seq before this
        # transform runs, so nothing to scale here regardless of decompose_obs.
        if not self.decompose_obs:
            return obs_seq

        # temporarily convert obs grid to the analysis grid
        # create the irregular obs grid
        if isinstance(c.grid, Grid1D):
            raise NotImplementedError("ScaleBandpass transform is not implemented for 1D grid.")
        obs_grid = IrregularGrid(c.grid.proj, obs_seq['x'], obs_seq['y'])

        # remove unwanted triangles in the obs grid
        tri_a = getattr(obs_grid.tri, 'a')
        tri_p = getattr(obs_grid.tri, 'p')
        tri_ratio = getattr(obs_grid.tri, 'ratio')
        max_a = np.quantile(tri_a, 0.999)
        max_p = np.quantile(tri_p, 0.99)
        msk = np.logical_or(tri_a > max_a, tri_p > max_p, tri_ratio < 0.3)

        # convert obs to analysis grid
        obs_grid = IrregularGrid(c.grid.proj, obs_seq['x'], obs_seq['y'], triangles=obs_grid.tri.triangles[~msk,:])
        obs_grid.set_destination_grid(c.grid)
        obs_fld = obs_grid.convert(obs_seq['obs'], is_vector=obs_rec.is_vector, method='nearest', coarse_grain=False)

        # pad voids with zeros, for convolution later
        mask = np.isnan(obs_fld)
        obs_fld[mask] = 0.0

        # get scale component on analysis grid
        obs_fld_new = get_scale_component(c.grid, obs_fld, self.character_length, c.iter)
        if obs_rec.is_vector:
            for i in range(2):
                obs_seq['obs'][i,...] = c.grid.interp(obs_fld_new[i,...], obs_seq['x'], obs_seq['y'], method='nearest')
        else:
            obs_seq['obs'] = c.grid.interp(obs_fld_new, obs_seq['x'], obs_seq['y'], method='nearest')

        return obs_seq

    def backward_obs(self, c, obs_rec, obs_seq):
        return obs_seq
