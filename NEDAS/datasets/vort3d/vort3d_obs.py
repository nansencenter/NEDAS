import numpy as np
from NEDAS.models.vort3d import Vort3DModel
from NEDAS.datasets.synthetic import SyntheticObs
from NEDAS.core.types import VarDesc

class Vort3DObs(SyntheticObs):
    """
    Synthetic obs for the vort3d model, mirroring vort2d_obs.Vort2DObs.

    Only the boundary-layer wind ('wind_b', i.e. ub/vb) is used for both the plain point-wind
    obs type and the vortex diagnostic obs operators below -- it's the layer most directly
    comparable to a real TC's near-surface circulation, and the same field already used for
    track/intensity/size diagnostics throughout the vort3d alignment-testbed and tutorial
    notebook work. vort3d's other state variables (free-atmosphere layers, theta, q, pstar) are
    NOT wired up here; add them the same way if/when needed.

    The vortex_position/vortex_intensity/vortex_size algorithms below are deliberately
    byte-identical to Vort2DObs's own (same box-summed-vorticity search, same wind-speed-based
    size/intensity definitions) -- this is not a coincidence or copy-paste laziness: the vort3d
    alignment-testbed scripts and the vort3d tutorial notebook already reuse
    Vort2DObs.vortex_position/vortex_size directly (via `Vort2DObs.__new__(Vort2DObs)`, since
    they're pure functions of (u, v) with no vort2d-specific state) for exactly this model, and
    all existing validated results were produced with this exact algorithm. Duplicating it here
    (rather than only NOW factoring out a shared module) keeps this dataset's own results
    consistent with everything already run, without touching the working vort2d_obs.py.
    """
    network_type: str
    obs_range: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        restart_dt = 6
        self.variables = {
            'wind_b': VarDesc(name='null', dtype='float', is_vector=True, dt=restart_dt, levels=np.array([0]), z_units='hPa', units='m/s'),
            'vortex_position': VarDesc(name='null', dtype='float', is_vector=True, dt=restart_dt, levels=np.array([0]), z_units='hPa', units='m'),
            'vortex_intensity': VarDesc(name='null', dtype='float', is_vector=False, dt=restart_dt, levels=np.array([0]), z_units='hPa', units='m/s'),
            'vortex_size':  VarDesc(name='null', dtype='float', is_vector=False, dt=restart_dt, levels=np.array([0]), z_units='hPa', units='m'),
        }

        self.obs_operator = {
            'vortex_position': self.get_vortex_position,
            'vortex_intensity': self.get_vortex_intensity,
            'vortex_size': self.get_vortex_size,
        }

    def generate_obs_network(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        name = kwargs['name']
        model = kwargs['model']
        assert isinstance(model, Vort3DModel)
        grid = kwargs['grid']

        # get truth vortex position, some network is vortex-following
        wind_b = self.get_wind_b(**{**kwargs, 'path': model.truth_dir})
        i, j = self.vortex_position(wind_b[0,...], wind_b[1,...])
        true_center_x, true_center_y = grid.x[j,i], grid.y[j,i]

        if name == 'wind_b':
            nobs = kwargs['nobs']
            if self.network_type == 'global':
                if nobs is None:
                    nobs = 1000
                y = np.random.uniform(grid.ymin, grid.ymax, nobs)
                x = np.random.uniform(grid.xmin, grid.xmax, nobs)

            elif self.network_type == 'targeted':
                if nobs is None:
                    nobs = 800
                x, y = [], []
                while len(x) < nobs:
                    x1 = np.random.uniform(true_center_x - self.obs_range, true_center_x + self.obs_range)
                    y1 = np.random.uniform(true_center_y - self.obs_range, true_center_y + self.obs_range)
                    dist = np.hypot(x1 - true_center_x, y1 - true_center_y)
                    if dist <= self.obs_range:
                        x.append(x1)
                        y.append(y1)
                x = np.array(x)
                y = np.array(y)

            else:
                raise ValueError('unknown network type: '+self.network_type)

            obs_seq = {'obs': np.full(nobs, np.nan),
                    't': np.full(nobs, kwargs['time']),
                    'z': np.zeros(nobs),
                    'y': y,
                    'x': x,
                    'err_std': np.ones(nobs) * kwargs['err']['std']
                    }

        elif name == 'vortex_position':
            obs_seq = {'obs': np.array([[np.nan, np.nan]]),
                    't': np.array([kwargs['time']]),
                    'z': np.array([0]),
                    'y': np.array([true_center_y]),
                    'x': np.array([true_center_x]),
                    'err_std': np.array([kwargs['err']['std']])
                    }

        elif name in ['vortex_intensity', 'vortex_size']:
            obs_seq = {'obs': np.array([np.nan]),
                    't': np.array([kwargs['time']]),
                    'z': np.array([0]),
                    'y': np.array([true_center_y]),
                    'x': np.array([true_center_x]),
                    'err_std': np.array([kwargs['err']['std']])
                    }

        else:
            raise ValueError('unknown obs variable: '+name)

        return obs_seq

    # utility functions for obs diagnostics -- identical to Vort2DObs's, see class docstring
    def vortex_position(self, u, v):
        ny, nx = u.shape

        # compute vorticity
        zeta = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1) - np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0)) / 2.0

        # search for max vorticity
        zmax = -999
        buff = 6
        center_i, center_j = None, None
        for j in range(buff, ny-buff):
            for i in range(buff, nx-buff):
                z = np.sum(zeta[j-buff:j+buff, i-buff:i+buff])
                if z > zmax:
                    zmax = z
                    center_i, center_j = i, j

        return center_i, center_j

    def vortex_intensity(self, u, v):
        return np.max(np.hypot(u, v))

    def vortex_size(self, u, v, center_i, center_j):
        wind = np.hypot(u, v)
        ny, nx = wind.shape

        nr = 30
        wind_min = 15
        wind_rad = np.zeros(nr)
        count_rad = np.zeros(nr)
        for j in range(-nr, nr+1):
            for i in range(-nr, nr+1):
                r = int(np.sqrt(i**2+j**2))
                if r < nr:
                    wind_rad[r] += wind[int(center_j+j)%ny, int(center_i+i)%nx]
                    count_rad[r] += 1
        wind_rad = wind_rad/count_rad

        if np.max(wind_rad)<wind_min or np.where(wind_rad>=wind_min)[0].size==0:
            Rsize = -1
        else:
            i1 = np.where(wind_rad>=wind_min)[0][-1] # last point with wind > wind_min
            if i1==nr-1:
                Rsize = i1
            else:
                Rsize = i1 + (wind_rad[i1] - wind_min) / (wind_rad[i1] - wind_rad[i1+1])

        return Rsize

    def get_wind_b(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        model = kwargs['model']
        assert isinstance(model, Vort3DModel), 'get_wind_b: ERROR: model must be an instance of Vort3DModel'
        grid = kwargs['grid']
        # read the boundary-layer wind field from truth
        model_wind_b = model.read_var(**{**kwargs, 'name':'wind', 'k': model.nz})  # k=nz is the boundary layer, per layer_names(nz)
        # convert to target grid
        model.grid.set_destination_grid(grid)
        wind_b = model.grid.convert(model_wind_b, is_vector=True)
        return wind_b

    def get_vortex_position(self, **kwargs):
        wind_b = self.get_wind_b(**kwargs)
        grid = kwargs['model'].grid
        center_i, center_j = self.vortex_position(wind_b[0,...], wind_b[1,...])
        obs_seq = np.zeros((2, 1), dtype='float')
        obs_seq[0,0] = grid.x[center_j, center_i]
        obs_seq[1,0] = grid.y[center_j, center_i]
        return obs_seq

    def get_vortex_intensity(self, **kwargs):
        wind_b = self.get_wind_b(**kwargs)
        Vmax = self.vortex_intensity(wind_b[0,...], wind_b[1,...])
        return np.array([Vmax])

    def get_vortex_size(self, **kwargs):
        wind_b = self.get_wind_b(**kwargs)
        dx = kwargs['model'].grid.dx
        center_i, center_j = self.vortex_position(wind_b[0,...], wind_b[1,...])
        Rsize = self.vortex_size(wind_b[0,...], wind_b[1,...], center_i, center_j)
        Rsize = Rsize * dx
        return np.array([Rsize])
