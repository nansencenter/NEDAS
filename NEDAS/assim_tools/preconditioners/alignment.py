import os
import types
import numpy as np
from scipy.ndimage import gaussian_filter
from NEDAS.core import Context, Preconditioner
from NEDAS.grid import Grid
from NEDAS.utils.optical_flow import OpticalFlow, warp

class AlignmentPreconditioner(Preconditioner):
    """
    Position-error correction, run before the assimilator (per Ravela, Emanuel & McLaughlin
    2007's "sequential solution": fix the state, solve for displacement -- this class; then fix
    the displacement, solve for the state residual -- the ordinary assimilator, unmodified).

    Per member: build two directly-comparable images on the OBSERVATION grid (image1 = this
    member's own H(x), already computed in c.obs.obs_prior; image2 = the observations
    themselves), run optical flow between them to get a displacement field, upsample that
    (coarse, obs-grid-resolution) displacement to the full analysis grid, and warp this
    member's prior state by it. Requires observations to sit on a regular, grid-aligned
    network (asserted below) -- this is what lets sparse point obs be reshaped into a
    comparable image at all; a genuinely irregular/scattered network would need a different
    approach (e.g. Ravela's own Hᵀ/elliptic-PDE alignment equation, not implemented here).

    Only meaningful for a single obs record / single state variable at a time (set via
    'variable' in preconditioner_def); state variables not matching are passed through
    untouched.

    'alpha' (relaxation factor, default 1.0 = full displacement applied): each member computes
    its own independent registration against a small (43x43), genuinely noisy obs-derived image,
    with no cross-member consistency constraint or smoothness regularization on the resulting
    displacement (unlike Ravela (2007)'s own cost function, which penalizes ||T|| and ||grad T||
    directly). 2026-07-15: at alpha=1.0 this let per-member registration noise inflate ensemble
    spread 3-5x above the SS baseline (cp10 N=20, spread ~1.3-1.5 for SS vs ~3-7 for
    AlignmentPreconditioner), degrading the full 60-cycle RMSE well below both SS and MSA despite
    single-cycle validation showing the mechanism itself (direction, warp sign) is correct.
    alpha<1 damps the applied warp to a fraction of the estimated displacement each cycle -- a
    relaxation-style mitigation, but a blunt one: it shrinks noise and genuine signal by the same
    proportion, so it can only trade off "less noise" against "less real correction" along one
    line, not actually separate the two. May be a single scalar (same alpha at every outer-loop
    iteration) or a list of length niter (one alpha per iteration, indexed by c.iter) -- useful
    with a multiscale (niter>1) outer loop, where finer scales (smaller character_length) likely
    have lower obs-image SNR after scale decomposition and may need heavier damping than the
    L-scale.

    'smooth_sigma' (grid points, in COARSE obs-grid units, default 0 = no smoothing): Gaussian-
    smooths the raw (dy,dx) displacement field on the coarse obs grid, before upsampling, with
    periodic ('wrap') boundary handling matching the model's doubly-periodic domain. Unlike
    alpha, this targets the noise directly: genuine displacement is expected to be spatially
    coherent (large-scale flow features), while per-member registration noise on a small, low-SNR
    image is expected to be much less spatially correlated -- smoothing suppresses the latter
    while mostly preserving the former, rather than shrinking both uniformly. Matches the spirit
    (if not the exact form) of Ravela (2007)'s own w2*||grad(T)|| smoothness penalty on the
    displacement field, which this project's optical-flow backends don't otherwise provide at
    this resolution.
    """
    variable: str
    alpha: float
    smooth_sigma: float

    def __init__(self, c: Context, **kwargs) -> None:
        super().__init__(c, **kwargs)
        self.variable = kwargs['variable']
        self.alpha = kwargs.get('alpha', 1.0)
        self.smooth_sigma = kwargs.get('smooth_sigma', 0.0)
        self.optical_flow = OpticalFlow(**kwargs.get('optical_flow', {}))

    def _find_obs_rec_id(self, c: Context) -> int:
        matches = [i for i, r in c.obs.info.records.items() if r.name == self.variable]
        assert len(matches) == 1, (
            f"AlignmentPreconditioner: expected exactly 1 obs record named '{self.variable}', "
            f"found {len(matches)}")
        return matches[0]

    def _obs_grid_coords(self, c: Context, obs_rec_id: int):
        seq = c.obs.obs_seq[obs_rec_id]
        unique_y = np.unique(seq['y'])
        unique_x = np.unique(seq['x'])
        nobs = seq['obs'].shape[-1]
        assert len(unique_y) * len(unique_x) == nobs, (
            "AlignmentPreconditioner requires a regular, grid-aligned obs network "
            f"(got {nobs} obs, {len(unique_y)}x{len(unique_x)} unique y/x values)")
        dy = unique_y[1] - unique_y[0]
        dx = unique_x[1] - unique_x[0]
        assert np.allclose(np.diff(unique_y), dy) and np.allclose(np.diff(unique_x), dx), (
            "AlignmentPreconditioner requires uniform obs spacing")
        return unique_y, unique_x, dy, dx

    def pre_assimilate(self, c: Context) -> None:
        obs_rec_id = self._find_obs_rec_id(c)
        unique_y, unique_x, dy, dx = self._obs_grid_coords(c, obs_rec_id)
        ny_obs, nx_obs = len(unique_y), len(unique_x)
        obs_grid = types.SimpleNamespace(nx=nx_obs, ny=ny_obs, dx=dx, dy=dy)

        # coarse (obs-resolution) grid, used only to upsample the displacement field onto the
        # full analysis grid via the same Grid.convert machinery used elsewhere in NEDAS
        # (e.g. alignment_updator.py's own displacement conversion), rather than a bespoke
        # interpolator -- consistent handling of projection/cyclic boundaries etc.
        x_coarse, y_coarse = np.meshgrid(unique_x, unique_y)
        coarse_grid = Grid(None, x_coarse, y_coarse, cyclic_dim='xy')
        coarse_grid.set_destination_grid(c.grid)

        image2 = c.obs.obs_seq[obs_rec_id]['obs'].reshape(ny_obs, nx_obs)

        for rec_id in c.state.rec_list[c.pid_rec]:
            rec = c.state.info.fields[rec_id]
            if rec.name != self.variable:
                continue
            for mem_id in c.mem_list[c.pid_mem]:
                key = (mem_id, rec_id)
                if key not in c.state.fields_prior:
                    continue
                obs_key = (mem_id, obs_rec_id)
                if obs_key not in c.obs.obs_prior:
                    # this rank doesn't own this member's obs_prior (rec-parallel distribution
                    # mismatch between state and obs) -- not handled here, see docstring
                    raise RuntimeError(
                        f"AlignmentPreconditioner: obs_prior missing for {obs_key} on this rank; "
                        "state/obs record distribution must match for this preconditioner")

                image1 = c.obs.obs_prior[obs_key].reshape(ny_obs, nx_obs)
                displace_coarse = self.optical_flow(obs_grid, image1, image2)  # (2, ny_obs, nx_obs), model-grid units
                if self.smooth_sigma > 0:
                    displace_coarse = np.array([
                        gaussian_filter(displace_coarse[0], sigma=self.smooth_sigma, mode='wrap'),
                        gaussian_filter(displace_coarse[1], sigma=self.smooth_sigma, mode='wrap'),
                    ])

                field = c.state.fields_prior[key]
                displace_full = coarse_grid.convert(displace_coarse, is_vector=True, method='linear')
                alpha = self.alpha[c.iter] if isinstance(self.alpha, (list, tuple)) else self.alpha
                u_full, v_full = alpha * displace_full[0], alpha * displace_full[1]

                self.displace[key] = np.array([u_full, v_full])
                # warp(x, u, v)[i,j] = x[i+v, j+u] -- to bring `field` (source of image1) INTO
                # alignment with image2, apply the NEGATED flow (verified empirically: for a
                # known np.roll(frame1, +S) giving flow=+S, only warp(frame1, -u, -v)
                # reconstructs frame2, not warp(frame1, +u, +v)).
                field_warped = warp(field, -u_full, -v_full)
                c.state.fields_prior[key] = field_warped

                if c.debug:
                    dbg_dir = os.path.join(c.config.work_dir, 'align_precond_debug')
                    os.makedirs(dbg_dir, exist_ok=True)
                    np.savez(os.path.join(dbg_dir, f'align_mem{mem_id}_rec{rec_id}.npz'),
                             image1=image1, image2=image2, displace_coarse=displace_coarse,
                             u_full=u_full, v_full=v_full,
                             field_before=field, field_after=field_warped)

        # fields_prior just changed for this variable -- obs_prior computed before this ran is
        # now stale; refresh so the assimilator sees obs priors consistent with the aligned state
        c.obs.prepare_obs_from_state(c, 'prior')
