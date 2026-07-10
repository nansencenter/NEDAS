import numpy as np
from NEDAS.core import Context, Updator
from NEDAS.utils.multiscale import get_remaining_scale_component
from .optical_flow import OpticalFlow, warp

class AlignmentUpdator(Updator):
    """Updator class with alignment technique.

    When interp_displaced_fields=False (default): displacement is applied by physically
    moving the model grid coordinates (Lagrangian, via model.displace if available).

    When interp_displaced_fields=True: displacement is applied by interpolating each
    field to the displaced grid positions without moving grid points themselves.
    """
    displace = {}

    def __init__(self, c: Context):
        super().__init__(c)
        alignment_opt = {k: v for k, v in c.config.updator_def.items() if k != 'type'}
        self.alignment_opt = alignment_opt
        self.interp_displaced_fields = alignment_opt.get('interp_displaced_fields', False)
        self.optical_flow = OpticalFlow(**alignment_opt.get('optical_flow', {}))

    def compute_increment(self, c: Context):
        """Compute optical flows from the prior/posterior state variable field pair."""
        c.print_1p("Compute alignment based on analysis increment of '"+self.alignment_opt.get('variable', 'unknown')+"'...\n")

        for rec_id in c.state.rec_list[c.pid_rec]:
            rec = c.state.info.fields[rec_id].asdict()
            model = c.models[rec['model_src']]
            if rec['name'] != self.alignment_opt['variable']:
                continue

            for mem_id in c.mem_list[c.pid_mem]:
                fld_prior = c.state.fields_prior[mem_id, rec_id]
                fld_post = c.state.fields_post[mem_id, rec_id]
                displace = self.optical_flow(c.grid, fld_prior, fld_post)
                self.displace[mem_id, rec['k']] = displace

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

        displace = self.displace[mem_id, rec['k']]
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
