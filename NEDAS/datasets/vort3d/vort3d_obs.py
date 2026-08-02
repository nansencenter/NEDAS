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

    The vortex_position/vortex_size algorithms below are deliberately byte-identical to
    Vort2DObs's own (same box-summed-vorticity search, same wind-speed-based size definition) --
    this is not a coincidence or copy-paste laziness: the vort3d alignment-testbed scripts and the
    vort3d tutorial notebook already reuse Vort2DObs.vortex_position/vortex_size directly (via
    `Vort2DObs.__new__(Vort2DObs)`, since they're pure functions of (u, v) with no vort2d-specific
    state) for exactly this model, and all existing validated results were produced with this
    exact algorithm. Duplicating it here (rather than only NOW factoring out a shared module)
    keeps this dataset's own results consistent with everything already run, without touching the
    working vort2d_obs.py.

    vortex_intensity is the one deliberate departure: Vort2DObs's own version is a domain-global
    `max(|wind|)`, which silently reports the wrong vortex's intensity whenever a second (spurious
    or genuinely distinct) vorticity feature anywhere in the domain happens to be windier than the
    tracked one -- observed concretely in a 2026-07-22 IC-perturbation sensitivity sweep, where a
    `Rmw_sprd`-perturbed member's compact vortex was sometimes out-competed in the *position*
    search too (a separate, already-known failure mode of the fixed-size search box), but a
    global-max intensity would have papered over that failure silently instead of surfacing it.
    Here, intensity is a *local* max within a box around the already-found vortex_position center
    (same box convention as vortex_size), so it reports the tracked vortex's own intensity or nan/
    garbage-but-visibly-so if the position search itself failed -- not some other feature's wind.
    """
    network_type: str
    obs_range: float
    core_bias_scale: float | None = None  # 2026-07-30, Yue: "make the network itself
    # core-weighted". None (default, backward compatible -- every experiment run before this
    # date keeps its exact original network) = uniform density over the obs_range disk, the
    # original 'targeted' behavior. Set to a length scale (e.g. comparable to Rmw) to instead
    # concentrate obs density near the vortex center: radius is drawn from a half-Gaussian with
    # this scale (rejected beyond obs_range) instead of uniformly over the disk area. Motivation:
    # uniform-over-disk sampling gives an EXPECTED core-hit count of only
    # nobs*(Rmw/obs_range)^2 -- e.g. ~0.4 obs within a 50km-radius core at nobs=100,
    # obs_range=800km -- so most draws put literally zero obs near the vortex core regardless of
    # nobs being "enough" in aggregate; core-weighting fixes that directly rather than requiring
    # nobs large enough to compensate statistically.
    core_bias_fraction: float = 1.0  # 2026-07-30, Yue: "the distribution of obs is now too
    # clustered in vortex core, tune the radial distribution down to more uniform version" --
    # pure core-biased sampling (fraction=1.0, the default once core_bias_scale is set) starves
    # the periphery of obs entirely, letting a single stray far-out obs dominate a wide area with
    # no local competition (confirmed 2026-07-30: an isolated 3-obs cluster ~650-690km out was
    # enough to make an edge pixel read HIGHER than the true, densely-observed vortex center,
    # which paradoxically gets the MOST diluted value of any point in the disk once hroi is
    # comparable to obs_range -- a geometry artifact, not a sign bug, see conversation). Setting
    # this < 1.0 draws that fraction of obs core-biased and the REST via the original
    # uniform-over-disk sampling, so the periphery keeps meaningful coverage too. Only used when
    # core_bias_scale is not None.
    zmin: float | None = None  # z-distribution range for the generic 'wind' obs type, in
    zmax: float | None = None  # `z_units` (default hPa) -- each obs draws its own z uniformly
    z_units: str = 'hPa'       # at random within [zmin, zmax]; zmin==zmax degenerates to a
    # single fixed level. 2026-07-25, replaces the earlier one-off 'wind_low' type (a second
    # hardcoded single-level proxy alongside 'wind_b') with a configurable range, since 'wind'
    # is already one of the model's own real state variables (model.variables['wind'], all
    # nz+1 levels) and NEDAS's generic state_to_obs option-1 pathway (core/obs.py) already does
    # correct vertical interpolation between levels given each obs's own z-coordinate -- no
    # custom obs_operator needed for 'wind' at all, unlike 'wind_b' (a dataset-level synthetic
    # proxy for one fixed level, NOT itself a model.variables entry, which is why it still needs
    # its own get_wind_b_obs below). zmin/zmax default to the boundary layer's own pressure
    # (i.e. a degenerate 'wind_b'-equivalent single level) if left unset.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # SyntheticObs.__init__ (the super().__init__ call above) auto-copies every
        # model.variables entry into self.variables, but ONLY if 'model_src' was itself part of
        # the kwargs used to CONSTRUCT this dataset -- true for a per-obs-record call, but NOT
        # true here: this project's dataset_def block (network_type/obs_range/zmin/zmax/z_units)
        # never includes model_src, so that auto-copy silently never fires in practice. Confirmed
        # 2026-07-25 the hard way: even after fixing the self.variables-overwrite bug below with
        # update() instead of reassignment, 'wind' obs still failed with the identical
        # "variable wind not defined in vort3d.dataset.variables" AssertionError, because there
        # was nothing to update() onto -- self.variables was empty at this point, not
        # auto-populated as assumed. Fixed by explicitly registering 'wind' from the vort3d model
        # directly (this dataset is vort3d-specific throughout anyway, per the Vort3DModel
        # isinstance asserts elsewhere in this file, so hardcoding the model name here is
        # consistent with the rest of the class, not a new assumption).
        # Guard: a generic smoke test (test_dataset_interface.py) constructs every
        # registered Dataset class against a bare Context() with no models registered at
        # all, so 'vort3d' is not guaranteed to be in self.c.models here -- only wire up
        # 'wind' when the model is actually present (true for any real vort3d config).
        if 'vort3d' in self.c.models:
            self.variables['wind'] = self.c.models['vort3d'].variables['wind']

        restart_dt = 6
        # NOTE: update(), not a wholesale `self.variables = {...}` reassignment -- would wipe out
        # the 'wind' entry just added above.
        self.variables.update({
            'wind_b': VarDesc(name='null', dtype='float', is_vector=True, dt=restart_dt, levels=np.array([0]), z_units='hPa', units='m/s'),
            'vortex_position': VarDesc(name='null', dtype='float', is_vector=True, dt=restart_dt, levels=np.array([0]), z_units='hPa', units='m'),
            'vortex_intensity': VarDesc(name='null', dtype='float', is_vector=False, dt=restart_dt, levels=np.array([0]), z_units='hPa', units='m/s'),
            'vortex_size':  VarDesc(name='null', dtype='float', is_vector=False, dt=restart_dt, levels=np.array([0]), z_units='hPa', units='m'),
        })

        self.obs_operator = {
            'wind_b': self.get_wind_b_obs,
            'vortex_position': self.get_vortex_position,
            'vortex_intensity': self.get_vortex_intensity,
            'vortex_size': self.get_vortex_size,
        }
        # 'wind' also has no obs_operator entry, deliberately -- option-1 in state_to_obs
        # (obs_name in model.variables) handles it without needing one.

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

        if name in ('wind_b', 'wind'):
            nobs = kwargs['nobs']
            if self.network_type == 'global':
                if nobs is None:
                    nobs = 1000
                y = np.random.uniform(grid.ymin, grid.ymax, nobs)
                x = np.random.uniform(grid.xmin, grid.xmax, nobs)

            elif self.network_type == 'targeted':
                if nobs is None:
                    nobs = 800

                def sample_uniform_disk(n):
                    xs, ys = [], []
                    while len(xs) < n:
                        x1 = np.random.uniform(true_center_x - self.obs_range, true_center_x + self.obs_range)
                        y1 = np.random.uniform(true_center_y - self.obs_range, true_center_y + self.obs_range)
                        dist = np.hypot(x1 - true_center_x, y1 - true_center_y)
                        if dist <= self.obs_range:
                            xs.append(x1)
                            ys.append(y1)
                    return xs, ys

                def sample_core_biased(n):
                    xs, ys = [], []
                    while len(xs) < n:
                        r = abs(np.random.normal(0, self.core_bias_scale))
                        if r <= self.obs_range:
                            theta = np.random.uniform(0, 2 * np.pi)
                            xs.append(true_center_x + r * np.cos(theta))
                            ys.append(true_center_y + r * np.sin(theta))
                    return xs, ys

                if self.core_bias_scale is None:
                    # original behavior: uniform density over the obs_range disk
                    x, y = sample_uniform_disk(nobs)
                else:
                    # mixture: core_bias_fraction of obs drawn core-weighted (half-Gaussian
                    # radius, see core_bias_scale's docstring), the rest uniform-over-disk so the
                    # periphery keeps real coverage too (see core_bias_fraction's docstring)
                    n_core = int(round(nobs * self.core_bias_fraction))
                    x_core, y_core = sample_core_biased(n_core)
                    x_unif, y_unif = sample_uniform_disk(nobs - n_core)
                    x, y = x_core + x_unif, y_core + y_unif
                x = np.array(x)
                y = np.array(y)

            else:
                raise ValueError('unknown network type: '+self.network_type)

            if name == 'wind':
                # z is physical pressure (Pa) -- same convention z_coords() itself uses, so it
                # lines up with the levels state_to_obs's vertical_interp brackets against.
                # zmin/zmax are given in z_units (default hPa); each obs draws its own z
                # independently within [zmin, zmax] (zmin==zmax gives a single fixed level).
                # Falls back to the boundary layer's own pressure (a degenerate single-level
                # 'wind_b'-equivalent) if zmin/zmax aren't explicitly set via dataset_def.
                if self.zmin is None or self.zmax is None:
                    zmin_pa = zmax_pa = model._layer_pressure[model.layer_names[-1]]
                else:
                    unit_scale = 100.0 if self.z_units == 'hPa' else 1.0  # hPa -> Pa
                    zmin_pa, zmax_pa = self.zmin * unit_scale, self.zmax * unit_scale
                z = np.random.uniform(zmin_pa, zmax_pa, nobs)
            else:
                z = np.zeros(nobs)  # 'wind_b': unused by its own custom obs_operator, kept as before

            obs_seq = {'obs': np.full(nobs, np.nan),
                    't': np.full(nobs, kwargs['time']),
                    'z': z,
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

    # utility functions for obs diagnostics -- vortex_position below now follows the
    # vorticity-centroid + first-guess-window approach standard in the TC-tracking literature
    # (e.g. Fang & Zhu 2019, https://www.mdpi.com/2073-4433/10/7/376) rather than Vort2DObs's
    # original discrete box-sum argmax; vortex_size stays byte-identical to Vort2DObs's, see
    # class docstring
    def vortex_position(self, u, v, first_guess=None, search_radius=20, vort_threshold_frac=0.5):
        """Vorticity-centroid center search, anchored to a first-guess position.

        Two problems with the original discrete box-summed-vorticity argmax (still used below
        only to *bootstrap* a first guess when none is given):
        (a) it can jump to a stronger, unrelated vorticity feature anywhere else in the domain --
            observed concretely in a 2026-07-22 IC-perturbation sensitivity sweep, where a few
            `Rmw_sprd`-perturbed members' compact vortices were passed over in favor of an
            unrelated Vbg-driven feature, producing 900+ km single-cycle position "jumps";
        (b) even when it stays on the right feature, picking a single integer grid cell as
            "the" center every cycle is sensitive to grid-scale vorticity noise, producing visible
            frame-to-frame jitter/zigzag in ensemble track spaghetti plots that isn't real vortex
            motion.

        Fix, following the standard TC-tracking approach: given `first_guess=(ci, cj)` (typically
        the previous timestep's own found center -- see find_track's chaining in
        vort3d/diagnostics.py), restrict the search to a `search_radius`-grid-cell window around
        it (fixes (a)), then take the vorticity-weighted CENTROID of cyclonic vorticity exceeding
        `vort_threshold_frac` of the window's peak (not a single-cell argmax) as the center
        (fixes (b) -- averaging over many grid cells largely cancels grid-scale noise, and
        thresholding first excludes the window's own weak background clutter from the centroid).
        With `first_guess=None` (e.g. the very first timestep of a track), a coarse whole-domain
        box-sum argmax bootstraps a first guess, which is then itself centroid-refined the same
        way."""
        ny, nx = u.shape

        # compute vorticity
        zeta = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1) - np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0)) / 2.0

        if first_guess is None:
            buff = 6
            zmax = -999
            center_i, center_j = None, None
            for j in range(buff, ny-buff):
                for i in range(buff, nx-buff):
                    z = np.sum(zeta[j-buff:j+buff, i-buff:i+buff])
                    if z > zmax:
                        zmax = z
                        center_i, center_j = i, j
            first_guess = (center_i, center_j)

        gi, gj = first_guess
        # x is periodic (see Core's own grid docstring); y is not
        i_idx = [i % nx for i in range(gi-search_radius, gi+search_radius+1)]
        j_idx = list(range(max(0, gj-search_radius), min(ny, gj+search_radius+1)))
        sub = zeta[np.ix_(j_idx, i_idx)]
        sub = np.clip(sub, 0, None)  # cyclonic (positive) vorticity only
        if sub.max() <= 0:
            return gi, gj  # nothing coherent in the window: stay at the first guess
        weight = np.where(sub >= vort_threshold_frac*sub.max(), sub, 0.0)
        jj, ii = np.meshgrid(j_idx, i_idx, indexing='ij')
        center_j = int(round(np.sum(weight*jj) / weight.sum()))
        center_i = int(round(np.sum(weight*ii) / weight.sum())) % nx

        return center_i, center_j

    def vortex_intensity(self, u, v, center_i, center_j, box=11):
        """local max wind speed within a box around the vortex center (NOT a domain-global
        max -- see class docstring for why)."""
        wind = np.hypot(u, v)
        ny, nx = wind.shape
        half = box // 2
        vmax = -999.
        for j in range(-half, half+1):
            for i in range(-half, half+1):
                vmax = max(vmax, wind[int(center_j+j)%ny, int(center_i+i)%nx])
        return vmax

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

    def get_wind_b_obs(self, **kwargs):
        """wind_b obs operator (registered in self.obs_operator, 2026-07-24 -- previously
        missing entirely: 'wind_b' matched neither a model.variables name (it's a dataset-
        level name, distinct from the model's own 'wind' state variable) nor an obs_operator
        entry, so state_to_obs raised "unable to obtain obs prior for 'wind_b'" for any
        actual wind_b assimilation attempt -- get_wind_b itself was only ever called
        internally by generate_obs_network and the vortex_* operators' own center-finding,
        which need the FULL 2D field, not point values).

        Interpolates the full boundary-layer wind field (from get_wind_b) to the scattered
        obs locations kwargs['x']/['y'] -- the same horizontal interpolation
        Obs.horizontal_interp does for the generic model.variables path, done directly here
        since only `grid` (not the full Context) is available inside an obs_operator call."""
        wind_b = self.get_wind_b(**kwargs)
        grid = kwargs['grid']
        obs_x, obs_y = np.array(kwargs['x']), np.array(kwargs['y'])
        f1 = grid.interp(wind_b[0, ...], obs_x, obs_y, method='linear')
        f2 = grid.interp(wind_b[1, ...], obs_x, obs_y, method='linear')
        return np.array([f1, f2])

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
        center_i, center_j = self.vortex_position(wind_b[0,...], wind_b[1,...])
        Vmax = self.vortex_intensity(wind_b[0,...], wind_b[1,...], center_i, center_j)
        return np.array([Vmax])

    def get_vortex_size(self, **kwargs):
        wind_b = self.get_wind_b(**kwargs)
        dx = kwargs['model'].grid.dx
        center_i, center_j = self.vortex_position(wind_b[0,...], wind_b[1,...])
        Rsize = self.vortex_size(wind_b[0,...], wind_b[1,...], center_i, center_j)
        Rsize = Rsize * dx
        return np.array([Rsize])
