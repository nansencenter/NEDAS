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
        # that iteration's scale-filtered field, as usual.
        # 'once_after_outer_loop': skip inflation during each iteration's assimilate() call;
        # schemes/filter.py::filter() instead calls final_post_inflation() once, after all
        # outer-loop iterations recombine into the full state. Matches the original design in
        # Ying (2019) where posterior inflation is domain-wide and computed/applied once on the
        # fully recombined analysis, not per scale. This can't be triggered from inside __call__
        # itself even on the last iteration: __call__ runs from Assimilator.assimilate(), which
        # happens BEFORE that same iteration's own Updator.update() call writes its increment to
        # the model's 'current' tag files -- final_post_inflation needs all iterations' updators
        # to have already run, so it has to be invoked from filter() after the whole outer loop.
        self.timing = timing

    def __call__(self, c: Context, flag: Literal['prior', 'post']) -> None:
        """
        Perform the covariance inflation method
        """
        if flag == 'prior':
            # cache the true cycle-start prior (iteration 0, before any of this cycle's outer-
            # loop DA) for final_post_inflation's own Desroziers stats later -- by the last
            # iteration, c.obs.obs_prior reflects the intermediate state from iterations
            # 0..iter-1, not the original prior. Not gated on self.prior (prior INFLATION being
            # enabled is a separate, orthogonal setting from timing) -- runs whenever this
            # Inflation instance uses once_after_outer_loop timing, regardless of self.prior.
            if self.timing == 'once_after_outer_loop' and c.iter == 0:
                c._cycle_obs_prior_full = {k: v.copy() for k, v in c.obs.obs_prior.items()}
            if self.prior:
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

    def final_post_inflation(self, c: Context) -> None:
        """
        Apply posterior inflation once, after all outer-loop iterations complete, to the full
        recombined state -- matching Ying (2019)'s design where the inflation factor is a single
        domain-wide Desroziers (2005) coefficient computed from the full state's obs-space
        statistics, applied once (not per scale-band iteration). Called by
        schemes/filter.py::filter() when self.timing == 'once_after_outer_loop', after the
        for-iter outer loop (see that timing's own docstring above for why this can't run from
        inside __call__).

        Uses the true cycle-start prior cached at iteration 0 (c._cycle_obs_prior_full, see
        __call__'s 'prior' branch) and the current c.obs.obs_post, which by this point reflects
        the fully recombined analysis (the last filter_iter() call recomputed it from the
        post-updator state). The model's 'current' tag files hold this same fully recombined
        state in native model space (assim_tools/updators/additive.py writes increments there
        each iteration), so this method reads/writes 'current' directly rather than going
        through c.state.fields_post, which only ever holds the last iteration's own
        scale-filtered field.
        """
        orig_obs_prior = c.obs.obs_prior
        c.obs.obs_prior = c._cycle_obs_prior_full
        self.adaptive_post_inflation(c)
        c.obs.obs_prior = orig_obs_prior
        coef = self.coef
        c.log_event(f"final posterior inflation coef={coef:.4f} (once, full recombined state)")

        # the last filter_iter()'s own updator.update() call already cleaned up its file locks
        # (core/updator.py::Updator.update(), c.comm.cleanup_file_locks() at the end) before
        # returning control here -- re-initialize locks for the SAME 'current' tag files this
        # method reads and writes below, otherwise acquire_file_lock's assertion fails with
        # "file lock ... not initialized" in offline (non-parallel-netcdf) io_mode. Reuses
        # c.updator's own init_all_file_locks (core/updator.py) rather than re-deriving the same
        # file list here -- c.updator is the same live instance filter_iter() already called
        # update() on, and init_all_file_locks is generic (keyed off c.mem_list/c.state.rec_list,
        # not anything updator-subclass-specific), so it's exactly the right file set.
        if c.config.io_mode == 'offline':
            c.updator.init_all_file_locks(c)

        for rec_id in c.state.rec_list[c.pid_rec]:
            rec = c.state.info.fields[rec_id]
            model = c.models[rec.model_src]

            sum_fld_pid = None
            for mem_id in c.mem_list[c.pid_mem]:
                fld = c.io.call_method(c, 'current', model.read_var, member=mem_id, **rec.asdict())
                if sum_fld_pid is None:
                    sum_fld_pid = np.zeros_like(fld)
                sum_fld_pid += fld
            sum_fld = c.comm_mem.allreduce(sum_fld_pid)
            mean_fld = sum_fld / c.nens

            for mem_id in c.mem_list[c.pid_mem]:
                fld = c.io.call_method(c, 'current', model.read_var, member=mem_id, **rec.asdict())
                fld_new = mean_fld + coef*(fld - mean_fld)
                c.io.call_method(c, 'current', model.write_var, fld_new, member=mem_id, **rec.asdict())
        c.comm.Barrier()
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

    @abstractmethod
    def adaptive_prior_inflation(self, c: Context):
        pass

    @abstractmethod
    def adaptive_post_inflation(self, c: Context):
        pass

    @abstractmethod
    def apply_inflation(self, c: Context, flag: Literal['prior', 'post']):
        pass
