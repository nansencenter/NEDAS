import os
import numpy as np
from NEDAS.core import Context, Updator
from NEDAS.utils.multiscale import get_remaining_scale_component
from NEDAS.utils.optical_flow import OpticalFlow, warp

def _vorticity(u, v):
    """Simple centered-difference vorticity, grid-index units (no dx scaling) -- used only as a
    scalar tracking image for optical flow, not a physically-scaled diagnostic."""
    return (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1) - np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0)) / 2.0


def _speed(u, v):
    """Wind speed |V| as a scalar tracking image."""
    return np.hypot(u, v)


def _deformation(u, v):
    """Total deformation sqrt(stretching^2 + shearing^2), same centered-difference stencil and
    grid-index units as _vorticity. Stretching D1 = du/dx - dv/dy, shearing D2 = dv/dx + du/dy."""
    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / 2.0
    dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / 2.0
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / 2.0
    dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / 2.0
    return np.hypot(dudx - dvdy, dvdx + dudy)


# scalar images derivable from a vector-valued alignment variable, selected via the
# updator_def 'vector_image' option (default 'vorticity', the historical behavior)
VECTOR_IMAGE_FUNCS = {
    'vorticity': _vorticity,
    'speed': _speed,
    'deformation': _deformation,
}


class AlignmentUpdator(Updator):
    """Updator class with alignment technique.

    When interp_displaced_fields=False (default): displacement is applied by physically
    moving the model grid coordinates (Lagrangian, via model.displace if available).

    When interp_displaced_fields=True: displacement is applied by interpolating each
    field to the displaced grid positions without moving grid points themselves.

    For a vector-valued `variable` (is_vector=True, e.g. a (u,v) wind field with no separate
    scalar diagnostic to align on): optical flow itself only operates on scalar images, so
    compute_increment derives displacement from a scalar image of the field rather than passing
    the raw (2,ny,nx) vector array to the optical-flow solver -- the resulting single
    displacement field is then applied to both vector components identically (update_files
    already handled this correctly; only the displacement *derivation* was vector-unaware).
    The scalar image is selectable via the updator_def 'vector_image' option:
    'vorticity' (default -- a compact single-signed monopole at a vortex core, the sharpest
    feature to track), 'speed' (|V| -- for a vortex an annulus with a central minimum, weaker
    aperture properties but no differencing noise), or 'deformation' (total deformation
    sqrt(stretching^2+shearing^2) -- highlights fronts/shear zones rather than rotation).
    """
    displace = {}

    def __init__(self, c: Context):
        super().__init__(c)
        alignment_opt = {k: v for k, v in c.config.updator_def.items() if k != 'type'}
        self.alignment_opt = alignment_opt
        self.interp_displaced_fields = alignment_opt.get('interp_displaced_fields', False)
        # optional reference level: when set, displacement is derived ONLY from the alignment
        # variable's record at this k, and that single displacement is applied to every level
        # of every state variable ("align by a given level"). When unset (default), each level
        # of a 3D alignment variable derives its own displacement ("align each level
        # separately"), and records at levels the alignment variable does not cover fall back
        # to the nearest available level's displacement (a 2D target like vort3d's pstar has
        # only k=0, so its single displacement is applied to the whole column).
        self.k_ref = alignment_opt.get('k', None)
        # scalar image derived from a vector-valued alignment variable (see class docstring)
        vector_image = alignment_opt.get('vector_image', 'vorticity')
        if vector_image not in VECTOR_IMAGE_FUNCS:
            raise ValueError(f"unknown vector_image '{vector_image}', "
                             f"choose from {sorted(VECTOR_IMAGE_FUNCS)}")
        self.vector_image_func = VECTOR_IMAGE_FUNCS[vector_image]
        self.optical_flow = OpticalFlow(**alignment_opt.get('optical_flow', {}))

    def _get_displace(self, mem_id, k):
        """Displacement for (member, level), with reference-level / nearest-level fallback."""
        if self.k_ref is not None:
            return self.displace[mem_id, self.k_ref]
        if (mem_id, k) in self.displace:
            return self.displace[mem_id, k]
        ks = [kk for (m, kk) in self.displace.keys() if m == mem_id]
        if not ks:
            raise KeyError(f"no displacement computed for member {mem_id} at any level")
        k_near = min(ks, key=lambda kk: abs(kk - k))
        return self.displace[mem_id, k_near]

    def compute_increment(self, c: Context):
        """Compute optical flows from the prior/posterior state variable field pair."""
        c.print_1p("Compute alignment based on analysis increment of '"+self.alignment_opt.get('variable', 'unknown')+"'...\n")

        for rec_id in c.state.rec_list[c.pid_rec]:
            rec = c.state.info.fields[rec_id].asdict()
            model = c.models[rec['model_src']]
            if rec['name'] != self.alignment_opt['variable']:
                continue
            if self.k_ref is not None and rec['k'] != self.k_ref:
                continue

            for mem_id in c.mem_list[c.pid_mem]:
                fld_prior = c.state.fields_prior[mem_id, rec_id]
                fld_post = c.state.fields_post[mem_id, rec_id]
                if rec['is_vector']:
                    img_prior = self.vector_image_func(fld_prior[0], fld_prior[1])
                    img_post = self.vector_image_func(fld_post[0], fld_post[1])
                else:
                    img_prior, img_post = fld_prior, fld_post
                displace = self.optical_flow(c.grid, img_prior, img_post)
                self.displace[mem_id, rec['k']] = displace

                # Diagnostic instrumentation (2026-07-14): dump the real fld_prior/fld_post/
                # displace actually seen by compute_increment, to measure how large the ACTUAL
                # prior-vs-EnKF-posterior displacement is in practice (as opposed to a
                # member-vs-truth proxy -- see qg_benchmark.md vault note). Gated on c.debug
                # (same switch as the rest of the codebase's debug-only diagnostics), no effect
                # on normal runs.
                if c.debug:
                    dbg_dir = os.path.join(c.config.work_dir, 'align_debug')
                    os.makedirs(dbg_dir, exist_ok=True)
                    np.savez(os.path.join(dbg_dir, f'align_mem{mem_id}_rec{rec_id}_iter{c.iter}.npz'),
                             fld_prior=fld_prior, fld_post=fld_post, displace=displace)

                if not self.interp_displaced_fields and hasattr(model, 'displace'):
                    # Lagrangian approach: physically move the model grid points
                    c.io.call_method(c, 'current', model.read_grid, member=mem_id, **rec)
                    c.grid.set_destination_grid(model.grid)
                    displace_m = c.grid.convert(displace, is_vector=True, method='linear')
                    c.io.call_method(c, 'current', getattr(model, 'displace'), displace_m[0,...], displace_m[1,...], member=mem_id, **rec)

        c.comm.Barrier()

    def update_files(self, c, mem_id, rec_id):
        """Apply displacement to model state variables.

        See Ying 2019 for details on the alignment technique.

        Ying (2019)'s design keeps scale components in separate arrays: the current
        scale's own posterior is finalized directly (no warp), and the displacement
        derived from it is applied only to the not-yet-processed finer-scale remainder,
        so already-finalized coarser scales are never re-displaced by a later
        iteration's (independently, noisily estimated) displacement.

        NEDAS stores only one recombined array per field, so var_prior (read below)
        already mixes finalized coarser scales, the current scale's old value, and the
        not-yet-processed remainder together. To reproduce Ying (2019)'s bookkeeping
        without a separate per-scale storage, we reconstruct the split on the analysis
        grid (bands are additive and telescope, see utils/multiscale.py), subtract it
        out at native model resolution to isolate the remainder, warp only that
        remainder, and leave the finalized bands untouched.
        """
        rec = c.state.info.fields[rec_id].asdict()
        model = c.models[rec['model_src']]
        fld_prior = c.state.fields_prior[mem_id, rec_id]
        fld_post = c.state.fields_post[mem_id, rec_id]

        var_prior = c.io.call_method(c, 'current', model.read_var, member=mem_id, **rec)
        c.grid.set_destination_grid(model.grid)

        if rec['is_vector']:
            fld_shape = var_prior.shape[1:]
        else:
            fld_shape = var_prior.shape

        displace = self._get_displace(mem_id, rec['k'])
        u_ana = displace[0,...] / c.grid.dx
        v_ana = displace[1,...] / c.grid.dx

        if self.interp_displaced_fields:
            # Interpolation approach: evaluate the remainder at displaced model-grid
            # positions, then add the finalized+current-band content unwarped — grid
            # points themselves do not move.

            # split the current full state (on the analysis grid) into: this iteration's
            # scale band (fld_prior, already computed by ScaleBandpass.forward_state),
            # the not-yet-processed remainder (finer bands), and by subtraction, the
            # already-finalized frozen bands (coarser, from earlier iterations)
            model.grid.set_destination_grid(c.grid)
            full_prior_ana = model.grid.convert(var_prior, is_vector=rec['is_vector'], method='linear', coarse_grain=True)
            remaining_ana = get_remaining_scale_component(c.grid, full_prior_ana, c.config.character_length, c.iter)
            frozen_ana = full_prior_ana - fld_prior - remaining_ana

            displace_m = c.grid.convert(displace, is_vector=True, method='linear')
            u, v = displace_m[0,...], displace_m[1,...]
            # taper_boundary is only relevant for models with a physical (non-cyclic) domain edge;
            # cyclic-domain models (e.g. qg.fortran) have no boundary to taper, so skip if absent.
            if hasattr(model, 'taper_boundary'):
                taper_boundary = getattr(model, 'taper_boundary')
                u = taper_boundary(u)
                v = taper_boundary(v)

            frozen_m = c.grid.convert(frozen_ana, is_vector=rec['is_vector'], method='linear')
            fld_post_m = c.grid.convert(fld_post, is_vector=rec['is_vector'], method='linear')
            # isolate the remainder at native model resolution (not just analysis-grid
            # resolution), by subtracting the (frozen + current-band-prior) content,
            # converted up from the analysis grid, from the actual native var_prior
            frozen_and_prior_m = c.grid.convert(frozen_ana + fld_prior, is_vector=rec['is_vector'], method='linear')
            remaining_native = var_prior - frozen_and_prior_m

            if fld_shape == model.grid.x.shape:
                if rec['is_vector']:
                    remaining_warp_x = model.grid.interp(remaining_native[0,...], model.grid.x-u, model.grid.y-v)
                    remaining_warp_y = model.grid.interp(remaining_native[1,...], model.grid.x-u, model.grid.y-v)
                    remaining_warp_m = np.array([remaining_warp_x, remaining_warp_y])
                else:
                    remaining_warp_m = model.grid.interp(remaining_native[...], model.grid.x-u, model.grid.y-v)
                var_post = frozen_m + fld_post_m + remaining_warp_m
            elif fld_shape == model.grid.x_elem.shape:
                u_elem = np.mean(u[...,model.grid.tri.triangles], axis=-1)
                v_elem = np.mean(v[...,model.grid.tri.triangles], axis=-1)
                remaining_warp_m = model.grid.interp(remaining_native, model.grid.x_elem-u_elem, model.grid.y_elem-v_elem)
                var_post = (remaining_warp_m
                            + np.mean(frozen_m[...,model.grid.tri.triangles], axis=-1)
                            + np.mean(fld_post_m[...,model.grid.tri.triangles], axis=-1))
            else:
                raise RuntimeError(f"mismatch in field prior {var_prior.shape} with remainder {remaining_native.shape}")

        else:
            # Grid-moving approach: grid already displaced via model.displace in compute_increment,
            # on the *whole* state file at once (model-specific, scale-unaware) -- this path does not
            # yet get the frozen-scale fix above, since separating already-finalized content from the
            # remainder would require compute_increment itself to warp only the remainder. Not exercised
            # by the qg_benchmark case study (qg.fortran has no model.displace); revisit if ever used
            # together with alignment.
            fld_prior_warp = fld_prior.copy()
            for ind in np.ndindex(fld_prior.shape[:-2]):
                fld_prior_warp[ind] = warp(fld_prior[ind], -u_ana, -v_ana)
            res_incr = fld_post - fld_prior_warp
            res_incr_m = c.grid.convert(res_incr, is_vector=rec['is_vector'], method='linear')
            if hasattr(model, 'displace'):
                if fld_shape == model.grid.x.shape:
                    var_post = var_prior + res_incr_m
                elif fld_shape == model.grid.x_elem.shape:
                    var_post = var_prior + np.mean(res_incr_m[...,model.grid.tri.triangles], axis=-1)
                else:
                    raise RuntimeError(f"mismatch in field prior {var_prior.shape} with residual increment {res_incr_m.shape}")
            else:
                new_var = c.grid.convert(fld_post, is_vector=rec['is_vector'], method='linear')
                if fld_shape == model.grid.x.shape:
                    var_post = new_var
                elif fld_shape == model.grid.x_elem.shape:
                    var_post = np.mean(new_var[...,model.grid.tri.triangles], axis=-1)
                else:
                    raise RuntimeError(f"mismatch in field prior {var_prior.shape} with posterior {new_var.shape}")

        ind = np.where(np.isnan(var_post))
        var_post[ind] = var_prior[ind]
        c.io.call_method(c, 'current', model.write_var, var_post, member=mem_id, **rec)
