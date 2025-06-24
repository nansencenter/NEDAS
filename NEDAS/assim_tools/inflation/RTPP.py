import numpy as np
from NEDAS.utils.progress import progress_bar
from NEDAS.assim_tools.inflation.base import Inflation

class RTPPInflation(Inflation):
    def adaptive_prior_inflation(self, c, state, obs):
        raise NotImplementedError("Relaxation method is only implemented for posterior ensemble")

    def adaptive_post_inflation(self, c, state, obs):
        """Adaptive covariance relaxation method (Ying and Zhang 2015, QJRMS)"""
        c.print_1p(">>> adaptive covariance relaxation:\n")
        stats = self.obs_space_stats(c, state, obs)
        if stats['total_nobs'] < 3:
            c.print_1p(f"Warning: insufficient nobs to establish statistics, setting self.coef=0")
            self.coef = 0.
        else:
            varb = stats['varb'] / stats['total_nobs']
            vara = stats['vara'] / stats['total_nobs']
            varo = stats['varo'] / stats['total_nobs']
            omb2 = stats['omb2'] / stats['total_nobs']
            omaamb = stats['omaamb'] / stats['total_nobs']
            amb2 = stats['amb2'] / stats['total_nobs']
            beta = np.sqrt(varb/vara)
            lamb = np.sqrt(max(0.0, (omb2-varo-amb2)/vara))
            c.print_1p(f"varb = {varb}, vara = {vara}, varo={varo}\n")
            c.print_1p(f"omb2 = {omb2}, omaamb = {omaamb}, amb2={amb2}\n")
            c.print_1p(f"beta = {beta}, lambda = {lamb}\n")
            if beta <= 1:
                self.coef = 0
            else:
                self.coef = (lamb - 1) / (beta - 1)
            if self.coef > 2:
                self.coef = 2
            if self.coef < -1:
                self.coef = -1

    def apply_inflation(self, c, state, flag):
        pid_mem_show = [p for p,lst in state.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in state.rec_list.items() if len(lst)>0][0]
        c.pid_show = pid_rec_show * c.nproc_mem + pid_mem_show
        c.print_1p(f'relaxing to prior ensemble perturbations with coef={self.coef}\n')

        ##process the fields, each processor goes through its own subset of
        ##mem_id,rec_id simultaneously
        nm = len(state.mem_list[c.pid_mem])
        nr = len(state.rec_list[c.pid_rec])

        for r, rec_id in enumerate(state.rec_list[c.pid_rec]):
            ##read the mean field with rec_id
            fld_prior_mean = state.read_field(state.prior_mean_file, c.grid.mask, 0, rec_id)
            fld_post_mean = state.read_field(state.post_mean_file, c.grid.mask, 0, rec_id)

            for m, mem_id in enumerate(state.mem_list[c.pid_mem]):
                if c.debug:
                    print(f"PID {c.pid:4}: relax_to_prior_perturb mem{mem_id+1:03}", flush=True)
                else:
                    c.print_1p(progress_bar(m*nr+r, nm*nr))
                ##inflate the ensemble perturbations by relaxing to prior perturbations
                fld_prior = state.fields_prior[mem_id, rec_id]
                fld_post = state.fields_post[mem_id, rec_id]
                fld_post = fld_post_mean + self.coef*(fld_prior - fld_prior_mean) + (1.-self.coef)*(fld_post - fld_post_mean)

        c.comm.Barrier()
        c.print_1p(' done.\n')
