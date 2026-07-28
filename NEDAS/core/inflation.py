from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from .context import Context

class Inflation(ABC):
    """
    Class for inflating the ensemble members (covariance inflation)
    """
    def __init__(self, coef: float=1.0,
                 adaptive: bool=False,
                 prior: bool=False, post: bool=False,
                 timing: Literal['per_iteration', 'once_after_outer_loop']='per_iteration'):
        self.coef = coef
        self.adaptive = adaptive
        self.prior = prior
        self.post = post
        # 'per_iteration' (default): apply inflation inside every outer-loop iteration, to
        # that iteration's scale-filtered field, via __call__ -> apply_inflation().
        #
        # 'once_after_outer_loop' (name kept for config/backward compatibility -- e.g.
        # qg_benchmark's configs already use this string; see 2026-07-28 revision note below):
        # inflation is applied ONCE, OUTSIDE the outer loop, on the true full-resolution state,
        # rather than per-iteration on each iteration's own scale-filtered field. The name is a
        # historical carryover from when only POSTERIOR inflation used this path -- "after" is
        # NOT generic enough to describe prior inflation under the same timing. The actual
        # invariant is "outside the outer loop": for flag='post' this means once AFTER all
        # outer-loop iterations complete (matching Ying (2019)'s domain-wide posterior
        # Desroziers coefficient, applied once on the fully recombined analysis, not per scale);
        # for flag='prior' this means once BEFORE any outer-loop iteration begins, on the
        # unmodified cycle-start forecast. schemes/filter.py::filter() is responsible for
        # calling apply_inflation_once() at the correct point for each flag -- see that
        # method's own docstring, and apply_inflation_once's, for the full mechanics.
        #
        # REVISION (2026-07-28, Yue): previously this was posterior-only (an ad hoc
        # final_post_inflation() method with its own hardcoded, multiplicative-only formula,
        # reused verbatim regardless of which Inflation subclass was active -- so e.g. RTPP's
        # own coef, which means something entirely different in RTPP's blend-with-prior formula
        # than in multiplicative's mean+coef*(pert) formula, was silently applied via the WRONG
        # formula whenever RTPP used this timing, actively deflating the ensemble instead of
        # relaxing it). Replaced with a proper abstract apply_inflation_once(), implemented per
        # subclass (mirrors apply_inflation's existing per-subclass pattern for the per-iteration
        # case), and prior inflation now gets its own once-before-outer-loop call instead of
        # being silently stuck on per-iteration application regardless of this timing setting.
        self.timing = timing

    def __call__(self, c: Context, flag: Literal['prior', 'post']) -> None:
        """
        Perform the covariance inflation method -- PER-ITERATION application only. Callers
        (core/assimilator.py) only invoke this when self.timing == 'per_iteration'; the 'once'
        (outside-the-outer-loop) case is scheduled directly by schemes/filter.py::filter() via
        apply_inflation_once() instead, both before (prior) and after (post) the outer loop.
        """
        if flag == 'prior' and self.prior:
            if self.adaptive:
                assert self.validate_obs_ens(c, c.obs.obs_prior), "obs.obs_prior is corrupted, cannot compute obs_space_stats for adaptive inflation."
                self.adaptive_prior_inflation(c)
            self.apply_inflation(c, flag)

        if flag == 'post' and self.post:
            if self.adaptive:
                assert self.validate_obs_ens(c, c.obs.obs_prior), "obs.obs_prior is corrupted, cannot compute obs_space_stats for adaptive inflation."
                assert self.validate_obs_ens(c, c.obs.obs_post), "obs.obs_post is corrupted, cannot compute obs_space_stats for adaptive inflation."
                self.adaptive_post_inflation(c)
            self.apply_inflation(c, flag)

    def _read_prior_field(self, c: Context, rec_id) -> tuple[dict, np.ndarray]:
        """Read the model's TRUE prior for every member of one record on this rank, directly
        from restart files on disk (tag='prior') rather than caching the full prior state in
        memory (2026-07-28, Yue: offline io_mode's whole design point is to avoid holding full
        ensemble states in memory -- restart files are the only efficient way). The offline io
        backend (io_backends/offline.py::OfflineIO.call_method) resolves tag='prior' to
        ens_init_dir's pre-staged restart files at c.time == c.config.time_start (the very
        first cycle, where there is no earlier forecast to read), or to the previous cycle's
        own forecast output otherwise -- the same resolution schemes/filter.py's own
        get_restart_dir() already uses for preprocess/postprocess/ensemble_forecast. Returns
        ({mem_id: fld}, ensemble_mean_fld), same shape as _read_current_field()."""
        rec = c.state.info.fields[rec_id]
        model = c.models[rec.model_src]
        flds = {}
        sum_fld_pid = None
        for mem_id in c.mem_list[c.pid_mem]:
            fld = c.io.call_method(c, 'prior', model.read_var, member=mem_id, **rec.asdict())
            flds[mem_id] = fld
            if sum_fld_pid is None:
                sum_fld_pid = np.zeros_like(fld)
            sum_fld_pid += fld
        sum_fld = c.comm_mem.allreduce(sum_fld_pid)
        mean_fld = sum_fld / c.nens
        return flds, mean_fld

    def _read_current_field(self, c: Context, rec_id) -> tuple[dict, np.ndarray]:
        """Read the model's 'current' tag field for every member of one record on this rank.
        Returns ({mem_id: fld}, ensemble_mean_fld). Shared I/O helper for
        apply_inflation_once() implementations, which all operate on 'current' directly rather
        than c.state.fields_{prior,post} (see apply_inflation_once's own docstring for why)."""
        rec = c.state.info.fields[rec_id]
        model = c.models[rec.model_src]
        flds = {}
        sum_fld_pid = None
        for mem_id in c.mem_list[c.pid_mem]:
            fld = c.io.call_method(c, 'current', model.read_var, member=mem_id, **rec.asdict())
            flds[mem_id] = fld
            if sum_fld_pid is None:
                sum_fld_pid = np.zeros_like(fld)
            sum_fld_pid += fld
        sum_fld = c.comm_mem.allreduce(sum_fld_pid)
        mean_fld = sum_fld / c.nens
        return flds, mean_fld

    def _field_variance(self, c: Context, flds: dict, mean_fld: np.ndarray) -> np.ndarray:
        """Ensemble variance (across the FULL ensemble, reduced over all ranks) of a set of
        per-member fields already read on this rank (e.g. from _read_prior_field() or
        _read_current_field()), given their pre-computed cross-rank ensemble mean. Shared by
        apply_inflation_once() implementations that need spread rather than the raw prior
        field itself (e.g. RTPS -- see assim_tools/inflation/RTPS.py)."""
        sum_sq_pid = None
        for fld in flds.values():
            d = fld - mean_fld
            if sum_sq_pid is None:
                sum_sq_pid = np.zeros_like(mean_fld)
            sum_sq_pid += d * d
        sum_sq = c.comm_mem.allreduce(sum_sq_pid)
        return sum_sq / (c.nens - 1)

    def _write_current_field(self, c: Context, rec_id, flds: dict) -> None:
        """Write {mem_id: fld} back to the model's 'current' tag for one record. Pairs with
        _read_current_field(); shared by apply_inflation_once() implementations."""
        rec = c.state.info.fields[rec_id]
        model = c.models[rec.model_src]
        for mem_id, fld in flds.items():
            c.io.call_method(c, 'current', model.write_var, fld, member=mem_id, **rec.asdict())

    def _init_current_file_locks(self, c: Context) -> None:
        """The last filter_iter()'s own updator.update() call already cleaned up its file locks
        (core/updator.py::Updator.update(), c.comm.cleanup_file_locks() at the end) -- or, for
        a prior-once call, no iteration has run yet at all -- either way, re-initialize locks
        for the SAME 'current' tag files apply_inflation_once() reads/writes, otherwise
        acquire_file_lock's assertion fails with "file lock ... not initialized" in offline
        (non-parallel-netcdf) io_mode. Reuses c.updator's own init_all_file_locks
        (core/updator.py) rather than re-deriving the same file list here -- generic (keyed off
        c.mem_list/c.state.rec_list, not anything updator-subclass-specific), so it's exactly
        the right file set regardless of when this is called."""
        if c.config.io_mode == 'offline':
            c.updator.init_all_file_locks(c)

    def _cleanup_current_file_locks(self, c: Context) -> None:
        if c.config.io_mode == 'offline':
            c.comm.cleanup_file_locks()

    def validate_obs_ens(self, c: Context, obs_ens: dict) -> bool:
        """ Check if the obs_ens has all member and records"""
        if isinstance(obs_ens, dict):
            for obs_rec_id in c.obs.obs_rec_list[c.pid_rec]:
                for mem_id in c.mem_list[c.pid_mem]:
                    if (mem_id, obs_rec_id) not in obs_ens:
                        return False
                    if not isinstance(obs_ens[mem_id, obs_rec_id], np.ndarray):
                        return False
            return True
        return False

    def obs_space_stats(self, c: Context):
        """observation-space statistics"""
        stats = {'total_nobs': 0,
                 'omb2': 0.0,  # obs-minus-background differences squared
                 'omaamb': 0.0,
                 'amb2': 0.0,  # analysis-minus-background diff squared
                 'varo': 0.0,  # obs err variance
                 'varb': 0.0,  # obs_prior (background) ensemble variances
                 'vara': 0.0,  # obs_post (analysis) ensemble variances
                }

        # Whether obs_post is available, agreed on once across all ranks in comm_mem.
        # c.obs.obs_post is populated all-or-nothing per rank (see
        # Assimilator.transpose_to_field_complete), so different ranks in the same
        # comm_mem group can locally disagree on its truthiness. Gating the
        # allreduce calls below on each rank's own local truthiness (as before) lets
        # ranks disagree on how many collective calls to make, which deadlocks MPI
        # instead of raising an error -- so decide it once, globally, and branch on
        # that everywhere instead.
        have_obs_post = c.comm_mem.allreduce(1 if c.obs.obs_post else 0) > 0

        # go through each obs record
        for r, obs_rec_id in enumerate(c.obs.obs_rec_list[c.pid_rec]):
            obs_rec = c.obs.info.records[obs_rec_id]
            nobs = obs_rec.nobs

            # 1. get ensemble mean obs_prior:
            if obs_rec.is_vector:
                nv = 2
                shape = (nv, nobs)
            else:
                nv = 1
                shape = (nobs,)

            # sum over all obs_prior_seq locally stored on pid
            sum_obs_prior_pid = np.zeros(shape)
            for mem_id in c.mem_list[c.pid_mem]:
                sum_obs_prior_pid += c.obs.obs_prior[mem_id, obs_rec_id]
            # sum over all obs_prior_seq on differnet pids to get the total sum
            sum_obs_prior = c.comm_mem.allreduce(sum_obs_prior_pid)
            mean_obs_prior = sum_obs_prior / c.nens
            mean_obs_post = None

            if have_obs_post:
                # sum over all obs_prior_seq locally stored on pid (zero if this
                # rank locally has no obs_post, so it still contributes to the
                # collective and stays in step with the other ranks)
                sum_obs_post_pid = np.zeros(shape)
                for mem_id in c.mem_list[c.pid_mem]:
                    sum_obs_post_pid += c.obs.obs_post.get((mem_id, obs_rec_id), np.zeros(shape))
                # sum over all obs_prior_seq on differnet pids to get the total sum
                sum_obs_post = c.comm_mem.allreduce(sum_obs_post_pid)
                mean_obs_post = sum_obs_post / c.nens

            # 2. get ensemble spread obs_prior:
            pert2_obs_prior_pid = np.zeros(shape)
            for mem_id in c.mem_list[c.pid_mem]:
                pert2_obs_prior_pid += (c.obs.obs_prior[mem_id, obs_rec_id] - mean_obs_prior)**2
            pert2_obs_prior = c.comm_mem.allreduce(pert2_obs_prior_pid)
            variance_obs_prior = pert2_obs_prior / (c.nens - 1)
            variance_obs_post = None

            if have_obs_post:
                pert2_obs_post_pid = np.zeros(shape)
                for mem_id in c.mem_list[c.pid_mem]:
                    if (mem_id, obs_rec_id) in c.obs.obs_post:
                        pert2_obs_post_pid += (c.obs.obs_post[mem_id, obs_rec_id] - mean_obs_post)**2
                pert2_obs_post = c.comm_mem.allreduce(pert2_obs_post_pid)
                variance_obs_post = pert2_obs_post / (c.nens - 1)

            obs_value = c.obs.obs_seq[obs_rec_id]['obs']
            stats['total_nobs'] += nv * nobs
            stats['omb2'] += np.sum((obs_value - mean_obs_prior)**2)
            stats['varo'] += np.sum(c.obs.obs_seq[obs_rec_id]['err_std']**2) * nv
            stats['varb'] += np.sum(variance_obs_prior)
            if have_obs_post and variance_obs_post is not None:
                stats['amb2'] += np.sum((mean_obs_post - mean_obs_prior)**2)
                stats['omaamb'] += np.sum((obs_value - mean_obs_post)*(mean_obs_post - mean_obs_prior))
                stats['vara'] += np.sum(variance_obs_post)
        return stats

    def relaxation_adaptive_coef(self, c: Context) -> float:
        """Adaptive relaxation coefficient (Ying and Zhang 2015, QJRMS) -- shared by RTPP and
        RTPS's own adaptive_post_inflation(), since the coefficient estimate itself doesn't
        depend on which relaxation formula (pointwise blend vs spread ratio) it's then used in.
        Ported 2026-07-28 directly from Yue's own original Fortran reference implementation
        (github.com/myying/PSU_WRF_EnKF, EnKF/src/enkf.f, relax_opt==1's adaptive branch) --
        corrects a bug in the previous NEDAS RTPP implementation, which computed:
            lamb = sqrt(max(0, (omb2-varo-amb2)/vara)); coef = (lamb-1)/(beta-1)
        an accidental reuse of MultiplicativeInflation's alternate 'omb2_amb2' POSTERIOR
        formula. The correct formula per the Fortran reference is:
            la = max(sqrt((omb2-varo)/varb), 1.0)     -- note: divides by varb (PRIOR
                                                            variance), no amb2 term at all
            beta = sqrt(varb/vara); ka = beta - 1
            coef = (la - 1.0) / ka
        Dividing by varb instead of vara matters: vara (posterior variance) is exactly the
        quantity a working relaxation scheme keeps from collapsing, so the old formula's
        vara-in-the-denominator was numerically fragile -- confirmed 2026-07-28 by a live
        36-cycle run where the coefficient estimate went unstable and hit NaN as vara shrank
        over cycles, corrupting the state on write. varb (prior/forecast variance) doesn't
        collapse the same way, so this formula is far more stable in practice.
        """
        stats = self.obs_space_stats(c)
        if stats['total_nobs'] < 3:
            if c.debug:
                c.log_event("insufficient nobs to establish statistics, setting coef=0", flag='warning')
            return 0.
        if stats['vara'] == 0 or stats['varb'] == 0:
            if c.debug:
                c.log_event("vara or varb == 0 detected, setting coef=0 (no relaxation)", flag='warning')
            return 0.
        varb = stats['varb'] / stats['total_nobs']
        vara = stats['vara'] / stats['total_nobs']
        varo = stats['varo'] / stats['total_nobs']
        omb2 = stats['omb2'] / stats['total_nobs']
        la = max(np.sqrt(max(0.0, (omb2 - varo) / varb)), 1.0)
        beta = np.sqrt(varb / vara)
        if c.debug:
            c.log_event(f"varb = {varb}, vara = {vara}, varo={varo}; omb2 = {omb2}; la = {la}, beta = {beta}", flag='stats')
        if beta <= 1:
            return 0.
        coef = (la - 1.0) / (beta - 1.0)
        if not np.isfinite(coef):
            if c.debug:
                c.log_event(f"non-finite relaxation coef (la={la}, beta={beta}), falling back to coef=0", flag='warning')
            return 0.
        c.message = f"varb = {varb}, vara = {vara}, varo={varo}; omb2 = {omb2}; la={la}, beta={beta}; coef = {coef}"
        return coef

    @abstractmethod
    def adaptive_prior_inflation(self, c: Context):
        pass

    @abstractmethod
    def adaptive_post_inflation(self, c: Context):
        pass

    @abstractmethod
    def apply_inflation(self, c: Context, flag: Literal['prior', 'post']):
        """Per-iteration application, on that iteration's own (possibly scale-filtered)
        c.state.fields_{prior,post}. Called by __call__(), i.e. only when
        self.timing == 'per_iteration'."""
        pass

    @abstractmethod
    def apply_inflation_once(self, c: Context, flag: Literal['prior', 'post']):
        """
        Apply inflation ONCE, outside the outer loop -- for flag='prior', called by
        schemes/filter.py::filter() once BEFORE any outer-loop iteration begins, on the true,
        unmodified cycle-start state; for flag='post', called once AFTER all outer-loop
        iterations complete, on the fully recombined state. Operates on the model's 'current'
        tag directly via _read_current_field()/_write_current_field() (not
        c.state.fields_{prior,post}, which are per-iteration and scale-filtered by that
        iteration's own transform_funcs, and don't survive past the iteration that created
        them) -- mirrors apply_inflation's role for the per-iteration case, but implemented per
        subclass since the required inputs differ: multiplicative inflation only needs the
        field currently being inflated (and its own ensemble mean); RTPP's blend-with-prior
        formula additionally needs the true prior, which for flag='post' is read via
        _read_prior_field() (restart files on disk, tag='prior' -- NOT cached in memory, per
        Yue's 2026-07-28 note that offline io_mode's whole point is to avoid holding full
        ensemble states in memory).
        """
        pass
