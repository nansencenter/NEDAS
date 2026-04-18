import numpy as np
from NEDAS.core.inflation import Inflation

class MultiplicativeInflation(Inflation):
    def adaptive_prior_inflation(self, c):
        """compute prior inflate coef by obs-space statistics (Desroziers et al. 2005)"""
        c.debug_message = "adaptive prior inflation"
        stats = self.obs_space_stats(c)
        if stats['total_nobs'] < 3:
            if c.debug:
                c.log_event(f"insufficient nobs to establish statistics, setting inflate_coef=1", flag='warning')
            self.coef = 1.
            return
        varb = stats['varb'] / stats['total_nobs']
        varo = stats['varo'] / stats['total_nobs']
        omb2 = stats['omb2'] / stats['total_nobs']
        if c.debug:
            c.log_event(f"varb = {varb}, varo={varo}; omb2 = {omb2}", flag='stats')
        self.coef = np.sqrt((omb2 - varo) / varb)
        c.message = f"varb = {varb}, varo={varo}; omb2 = {omb2}; coef = {self.coef}"

    def adaptive_post_inflation(self, c):
        """compute posterior inflate coef by obs-space statistics (Desroziers et al. 2005) """
        c.debug_message = "adaptive posterior inflation"
        stats = self.obs_space_stats(c)
        if stats['total_nobs'] < 3:
            if c.debug:
                c.log_event(f"insufficient nobs to establish statistics, setting inflate_coef=1", flag='warning')
            self.coef = 1.
            return
        if stats['vara'] == 0:
            if c.debug:
                c.log_event(f"vara=0 detected, skipping with coef=1 (no inflation)", flag='warning')
            self.coef = 1.
            return
        varb = stats['varb'] / stats['total_nobs']
        vara = stats['vara'] / stats['total_nobs']
        varo = stats['varo'] / stats['total_nobs']
        omb2 = stats['omb2'] / stats['total_nobs']
        omaamb = stats['omaamb'] / stats['total_nobs']
        amb2 = stats['amb2'] / stats['total_nobs']
        if c.debug:
            c.log_event(f"varb = {varb}, vara = {vara}, varo={varo}; omb2 = {omb2}, omaamb = {omaamb}, amb2={amb2}", flag='stats')
        # self.coef = np.sqrt(omaamb/vara)
        ratio = (omb2-varo-amb2)/vara
        if ratio < 0:
            self.coef = 1.0
            return
        self.coef = np.sqrt(ratio)
        c.message = f"varb = {varb}, vara = {vara}, varo={varo}; coef = {self.coef}"

    def apply_inflation(self, c, flag):
        if flag not in ['prior', 'post']:
            raise ValueError(f"Unknown flag {flag}, should be prior or post")
        fields = getattr(c.state, f"fields_{flag}")

        pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in c.state.rec_list.items() if len(lst)>0][0]
        c.pid_show = pid_rec_show * c.config.nproc_mem + pid_mem_show

        c.debug_message = f'inflating {flag} ensemble with multiplicative coef={self.coef}'

        # process the fields, each processor goes through its own subset of
        # mem_id,rec_id simultaneously
        nm = len(c.mem_list[c.pid_mem])
        nr = len(c.state.rec_list[c.pid_rec])
        c.total_tasks = nm * nr
        for r, rec_id in enumerate(c.state.rec_list[c.pid_rec]):

            # read the mean field with rec_id
            #c.io.read_field()
            fields_mean = c.io.read_field(c, f"{flag}_mean", rec_id, mem_id=0)
            for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
                c.debug_message = f"inflating mem{mem_id+1:03}"
                c.current_task = m*nr+r

                # inflate the ensemble perturbations by coef
                fields[mem_id, rec_id] = self.coef*(fields[mem_id, rec_id] - fields_mean) + fields_mean

        c.comm.Barrier()
