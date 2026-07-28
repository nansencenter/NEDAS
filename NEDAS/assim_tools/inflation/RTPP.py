import numpy as np
from NEDAS.core import Context, Inflation

class RTPPInflation(Inflation):
    def adaptive_prior_inflation(self, c: Context):
        raise NotImplementedError("Relaxation method is only implemented for posterior ensemble")

    def adaptive_post_inflation(self, c: Context):
        """Adaptive covariance relaxation coefficient (Ying and Zhang 2015, QJRMS) -- see
        Inflation.relaxation_adaptive_coef()'s own docstring for the corrected formula
        (2026-07-28) and why the previous vara-denominator version here was numerically
        fragile."""
        self.coef = self.relaxation_adaptive_coef(c)

    def apply_inflation(self, c: Context, flag: str):
        pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in c.state.rec_list.items() if len(lst)>0][0]
        c.pid_show = pid_rec_show * c.config.nproc_mem + pid_mem_show

        if c.debug:
            c.log_event(f'relaxing to prior ensemble perturbations with coef={self.coef}', flag='info')

        # process the fields, each processor goes through its own subset of
        # mem_id,rec_id simultaneously
        nm = len(c.mem_list[c.pid_mem])
        nr = len(c.state.rec_list[c.pid_rec])
        c.total_tasks = nm*nr
        for r, rec_id in enumerate(c.state.rec_list[c.pid_rec]):
            # read the mean field with rec_id
            fld_prior_mean = c.io.read_field(c, 'prior_mean', rec_id, mem_id=0)
            fld_post_mean = c.io.read_field(c, 'post_mean', rec_id, mem_id=0)

            for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
                c.debug_message = f"relax_to_prior_perturb mem{mem_id+1:03}"
                c.current_task = m*nr+r

                # inflate the ensemble perturbations by relaxing to prior perturbations
                fld_prior = c.state.fields_prior[mem_id, rec_id]
                fld_post = c.state.fields_post[mem_id, rec_id]
                fld_post = fld_post_mean + self.coef*(fld_prior - fld_prior_mean) + (1.-self.coef)*(fld_post - fld_post_mean)

        c.comm.Barrier()

    def apply_inflation_once(self, c: Context, flag: str):
        """Once-outside-the-outer-loop application (see Inflation.apply_inflation_once's own
        docstring). flag='prior' is unsupported here for the same reason as
        adaptive_prior_inflation -- RTPP's relaxation is inherently a posterior-vs-prior blend,
        there's no equivalent "relax to X" operation for a lone prior field. flag='post'
        implements the real RTPP blend on the true full-resolution state: reads the true prior
        via restart files (_read_prior_field(), tag='prior' -- correct even at the very first
        cycle, see io_backends/offline.py's own ens_init_dir handling there) and the fully
        recombined posterior via 'current' (_read_current_field()), blends, writes back to
        'current'. Fixes a 2026-07-28 bug where this timing previously ran through the base
        class's old final_post_inflation(), which applied a hardcoded MULTIPLICATIVE formula
        (mean + coef*(fld-mean)) regardless of which Inflation subclass was active -- for RTPP,
        whose coef is always in [0,1] (a relaxation weight, not a multiplicative factor), that
        formula actively DEFLATED the ensemble instead of relaxing it toward the prior."""
        if flag == 'prior':
            raise NotImplementedError("Relaxation method is only implemented for posterior ensemble")

        if self.adaptive:
            orig_obs_prior = c.obs.obs_prior
            c.obs.obs_prior = c._cycle_obs_prior_full
            self.adaptive_post_inflation(c)
            c.obs.obs_prior = orig_obs_prior
        coef = self.coef
        c.log_event(f"relaxing to prior ensemble perturbations (once, full recombined state), coef={coef:.4f}")

        self._init_current_file_locks(c)
        for rec_id in c.state.rec_list[c.pid_rec]:
            prior_flds, prior_mean = self._read_prior_field(c, rec_id)
            post_flds, post_mean = self._read_current_field(c, rec_id)
            new_flds = {}
            for mem_id in c.mem_list[c.pid_mem]:
                new_flds[mem_id] = (post_mean + coef * (prior_flds[mem_id] - prior_mean)
                                     + (1. - coef) * (post_flds[mem_id] - post_mean))
            self._write_current_field(c, rec_id, new_flds)
        c.comm.Barrier()
        self._cleanup_current_file_locks(c)
