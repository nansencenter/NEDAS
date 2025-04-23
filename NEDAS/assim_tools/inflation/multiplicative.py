import numpy as np
from NEDAS.utils.progress import progress_bar
from NEDAS.assim_tools.inflation.base import Inflation

class MultiplicativeInflation(Inflation):
    def adaptive_prior_inflation(self, c, state, obs):
        """compute prior inflate coef by obs-space statistics (Desroziers et al. 2005)"""
        c.print_1p(">>> adaptive prior inflation:\n")
        stats = self.obs_space_stats(c, state, obs)
        if stats['total_nobs'] < 3:
            c.print_1p(f"Warning: insufficient nobs to establish statistics, setting inflate_coef=1")
            self.coef = 1.
        else:
            varb = stats['varb'] / stats['total_nobs']
            varo = stats['varo'] / stats['total_nobs']
            omb2 = stats['omb2'] / stats['total_nobs']
            c.print_1p(f"varb = {varb}, varo={varo}\n")
            c.print_1p(f"omb2 = {omb2}\n")
            self.coef = np.sqrt((omb2 - varo) / varb)

    def adaptive_post_inflation(self, c, state, obs):
        """compute posterior inflate coef by obs-space statistics (Desroziers et al. 2005) """
        c.print_1p(">>> adaptive posterior inflation:\n")
        stats = self.obs_space_stats(c, state, obs)
        if stats['total_nobs'] < 3:
            c.print_1p(f"Warning: insufficient nobs to establish statistics, setting inflate_coef=1")
            self.coef = 1.
        else:
            varb = stats['varb'] / stats['total_nobs']
            vara = stats['vara'] / stats['total_nobs']
            varo = stats['varo'] / stats['total_nobs']
            omb2 = stats['omb2'] / stats['total_nobs']
            omaamb = stats['omaamb'] / stats['total_nobs']
            amb2 = stats['amb2'] / stats['total_nobs']
            c.print_1p(f"varb = {varb}, vara = {vara}, varo={varo}\n")
            c.print_1p(f"omb2 = {omb2}, omaamb = {omaamb}, amb2={amb2}\n")
            # self.coef = np.sqrt(omaamb/vara)
            self.coef = np.sqrt((omb2-varo-amb2)/vara)

    def apply_inflation(self, c, state, flag):
        if flag == 'prior':
            mean_file = state.prior_mean_file
            fields = state.fields_prior
        elif flag == 'posterior':
            mean_file = state.post_mean_file
            fields = state.fields_post
        else:
            raise ValueError(f"Unknown flag {flag}, should be prior or posterior")

        pid_mem_show = [p for p,lst in state.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in state.rec_list.items() if len(lst)>0][0]
        c.pid_show = pid_rec_show * c.nproc_mem + pid_mem_show

        c.print_1p(f'>>> inflating {flag} ensemble with multiplicative coef={self.coef}\n')

        ##process the fields, each processor goes through its own subset of
        ##mem_id,rec_id simultaneously
        nm = len(state.mem_list[c.pid_mem])
        nr = len(state.rec_list[c.pid_rec])

        for r, rec_id in enumerate(state.rec_list[c.pid_rec]):

            ##read the mean field with rec_id
            fields_mean = state.read_field(mean_file, c.grid.mask, 0, rec_id)
            for m, mem_id in enumerate(state.mem_list[c.pid_mem]):
                if c.debug:
                    print(f"PID {c.pid:4}: inflating mem{mem_id+1:03}", flush=True)
                else:
                    c.print_1p(progress_bar(m*nr+r, nm*nr))

                ##inflate the ensemble perturbations by coef
                fields[mem_id, rec_id] = self.coef*(fields[mem_id, rec_id] - fields_mean) + fields_mean

        c.comm.Barrier()
        c.print_1p(' done.\n')
