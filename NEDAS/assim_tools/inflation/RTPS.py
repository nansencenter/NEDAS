import numpy as np
from NEDAS.core import Context, Inflation

class RTPSInflation(Inflation):
    """Relaxation To Prior Spread (Whitaker & Hamill 2012, MWR; also compared against RTPP in
    Ying & Zhang 2015, QJRMS). Unlike RTPP, which blends prior and posterior PERTURBATIONS
    pointwise (fld_prior(i,j) vs fld_post(i,j) at the same grid index), RTPS only uses each
    ensemble's local STANDARD DEVIATION (a per-point scalar statistic, computed independently
    within the prior ensemble and within the posterior ensemble) -- it never mixes a prior
    field VALUE with a posterior field VALUE at the same point. This matters specifically for
    alignment-based updators (MSA/HornSchunck): the analysis WARPS the field, so grid point
    (i,j) in the prior and grid point (i,j) in the post-alignment posterior are not the same
    physical air parcel once a nontrivial warp has been applied -- RTPP's pointwise blend
    reintroduces spatially incoherent values exactly where the warp was strongest (confirmed
    2026-07-28: a real NaN blow-up, MSA + RTPP coef=0.8, day3fps window -- pointwise
    prior/posterior differences up to 38 m/s wind / 12 K theta at single grid points, despite
    unremarkable domain-mean statistics). RTPS's spread-only formula has no such pointwise
    correspondence requirement, so it should be structurally immune to that specific failure
    mode -- implemented 2026-07-28, Yue, to test that directly.

    x_a_new(i,j) = post_mean(i,j) + factor(i,j) * (x_a(i,j) - post_mean(i,j))
    factor(i,j) = 1 + coef * (sigma_b(i,j) - sigma_a(i,j)) / sigma_a(i,j)

    where sigma_b/sigma_a are the prior/posterior ensemble standard deviations at each point,
    coef in [0,1] (0 = no relaxation, 1 = fully restore prior spread).
    """
    def adaptive_prior_inflation(self, c: Context):
        raise NotImplementedError("Relaxation method is only implemented for posterior ensemble")

    def adaptive_post_inflation(self, c: Context):
        """Adaptive covariance relaxation coefficient (Ying and Zhang 2015, QJRMS) -- shares
        RTPPInflation's own estimate (Inflation.relaxation_adaptive_coef(), corrected
        2026-07-28) since the coefficient estimate itself doesn't depend on which state-space
        relaxation formula (pointwise blend vs spread-only) is then used to apply it."""
        self.coef = self.relaxation_adaptive_coef(c)

    def _rtps_factor(self, prior_std: np.ndarray, post_std: np.ndarray, coef: float) -> np.ndarray:
        factor = np.ones_like(post_std)
        mask = post_std > 0
        factor[mask] = 1.0 + coef * (prior_std[mask] - post_std[mask]) / post_std[mask]
        return factor

    def apply_inflation(self, c: Context, flag: str):
        """Per-iteration application, on that iteration's own (possibly scale-filtered)
        c.state.fields_{prior,post} -- mirrors RTPPInflation.apply_inflation's own role/
        structure, spread-ratio formula instead of pointwise blend."""
        if flag != 'post':
            raise NotImplementedError("Relaxation method is only implemented for posterior ensemble")
        pid_mem_show = [p for p, lst in c.mem_list.items() if len(lst) > 0][0]
        pid_rec_show = [p for p, lst in c.state.rec_list.items() if len(lst) > 0][0]
        c.pid_show = pid_rec_show * c.config.nproc_mem + pid_mem_show

        if c.debug:
            c.log_event(f'relaxing to prior ensemble spread with coef={self.coef}', flag='info')

        nm = len(c.mem_list[c.pid_mem])
        nr = len(c.state.rec_list[c.pid_rec])
        c.total_tasks = nm * nr
        for r, rec_id in enumerate(c.state.rec_list[c.pid_rec]):
            fld_prior_mean = c.io.read_field(c, 'prior_mean', rec_id, mem_id=0)
            fld_post_mean = c.io.read_field(c, 'post_mean', rec_id, mem_id=0)
            prior_flds = {m: c.state.fields_prior[m, rec_id] for m in c.mem_list[c.pid_mem]}
            post_flds = {m: c.state.fields_post[m, rec_id] for m in c.mem_list[c.pid_mem]}
            prior_var = self._field_variance(c, prior_flds, fld_prior_mean)
            post_var = self._field_variance(c, post_flds, fld_post_mean)
            factor = self._rtps_factor(np.sqrt(prior_var), np.sqrt(post_var), self.coef)

            for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
                c.debug_message = f"relax_to_prior_spread mem{mem_id+1:03}"
                c.current_task = m * nr + r
                fld_post = c.state.fields_post[mem_id, rec_id]
                c.state.fields_post[mem_id, rec_id] = fld_post_mean + factor * (fld_post - fld_post_mean)

        c.comm.Barrier()

    def apply_inflation_once(self, c: Context, flag: str):
        """Once-outside-the-outer-loop application (see Inflation.apply_inflation_once's own
        docstring). flag='prior' unsupported for the same reason as adaptive_prior_inflation.
        flag='post': reads prior/posterior ensembles via restart files (_read_prior_field(),
        _read_current_field() -- correct even at the very first cycle, see
        io_backends/offline.py), computes each ensemble's own local standard deviation
        (_field_variance()), and rescales the posterior perturbations by the spread ratio --
        no pointwise prior/posterior VALUE blending at all, see class docstring for why that
        matters for alignment-warped fields."""
        if flag == 'prior':
            raise NotImplementedError("Relaxation method is only implemented for posterior ensemble")

        if self.adaptive:
            orig_obs_prior = c.obs.obs_prior
            c.obs.obs_prior = c._cycle_obs_prior_full
            self.adaptive_post_inflation(c)
            c.obs.obs_prior = orig_obs_prior
        coef = self.coef
        c.log_event(f"relaxing to prior ensemble spread (once, full recombined state), coef={coef:.4f}")

        self._init_current_file_locks(c)
        for rec_id in c.state.rec_list[c.pid_rec]:
            prior_flds, prior_mean = self._read_prior_field(c, rec_id)
            post_flds, post_mean = self._read_current_field(c, rec_id)
            prior_std = np.sqrt(self._field_variance(c, prior_flds, prior_mean))
            post_std = np.sqrt(self._field_variance(c, post_flds, post_mean))
            factor = self._rtps_factor(prior_std, post_std, coef)
            new_flds = {mem_id: post_mean + factor * (fld - post_mean) for mem_id, fld in post_flds.items()}
            self._write_current_field(c, rec_id, new_flds)
        c.comm.Barrier()
        self._cleanup_current_file_locks(c)
