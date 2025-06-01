from abc import ABC, abstractmethod
import numpy as np

class Inflation(ABC):
    """
    Class for inflating the ensemble members (covariance inflation)
    """
    def __init__(self, coef: float=1.0,
                 adaptive: bool=False,
                 prior: bool=False, posterior: bool=False):
        self.coef = coef
        self.adaptive = adaptive
        self.prior = prior
        self.posterior = posterior

    def __call__(self, c, state, obs, flag):
        """
        Perform the covariance inflation method
        Input:
        -c: config object
        -state: State object with fields_prior, field_post
        -obs: Obs object with obs_seq, obs_prior_seq, obs_post_seq
        -flag: 'prior' or 'posterior'
        """
        if flag == 'prior' and self.prior:
            if self.adaptive:
                self.adaptive_prior_inflation(c, state, obs)
            self.apply_inflation(c, state, flag)

        if flag == 'posterior' and self.posterior:
            if self.adaptive:
                self.adaptive_post_inflation(c, state, obs)
            self.apply_inflation(c, state, flag)

    def obs_space_stats(self, c, state, obs):
        """observation-space statistics"""
        stats = {'total_nobs': 0,
                 'omb2': 0.0,  ##obs-minus-background differences squared
                 'omaamb': 0.0,
                 'amb2': 0.0,  ##analysis-minus-background diff squared
                 'varo': 0.0,  ##obs err variance
                 'varb': 0.0,  ##obs_prior (background) ensemble variances
                 'vara': 0.0,  ##obs_post (analysis) ensemble variances
                }

        ##go through each obs record
        for r, obs_rec_id in enumerate(obs.obs_rec_list[c.pid_rec]):
            obs_rec = obs.info['records'][obs_rec_id]
            nobs = obs_rec['nobs']

            ##1. get ensemble mean obs_prior:
            if obs_rec['is_vector']:
                nv = 2
                shape = (nv, nobs)
            else:
                nv = 1
                shape = (nobs,)

            ##sum over all obs_prior_seq locally stored on pid
            sum_obs_prior_pid = np.zeros(shape)
            for mem_id in state.mem_list[c.pid_mem]:
                sum_obs_prior_pid += obs.obs_prior_seq[mem_id, obs_rec_id]
            ##sum over all obs_prior_seq on differnet pids to get the total sum
            sum_obs_prior = c.comm_mem.allreduce(sum_obs_prior_pid)
            mean_obs_prior = sum_obs_prior / c.nens

            if obs.obs_post_seq:
                ##sum over all obs_prior_seq locally stored on pid
                sum_obs_post_pid = np.zeros(shape)
                for mem_id in state.mem_list[c.pid_mem]:
                    sum_obs_post_pid += obs.obs_post_seq[mem_id, obs_rec_id]
                ##sum over all obs_prior_seq on differnet pids to get the total sum
                sum_obs_post = c.comm_mem.allreduce(sum_obs_post_pid)
                mean_obs_post = sum_obs_post / c.nens

            ##2. get ensemble spread obs_prior:
            pert2_obs_prior_pid = np.zeros(shape)
            for mem_id in state.mem_list[c.pid_mem]:
                pert2_obs_prior_pid += (obs.obs_prior_seq[mem_id, obs_rec_id] - mean_obs_prior)**2
            pert2_obs_prior = c.comm_mem.allreduce(pert2_obs_prior_pid)
            variance_obs_prior = pert2_obs_prior / (c.nens - 1)

            if obs.obs_post_seq:
                pert2_obs_post_pid = np.zeros(shape)
                for mem_id in state.mem_list[c.pid_mem]:
                    pert2_obs_post_pid += (obs.obs_post_seq[mem_id, obs_rec_id] - mean_obs_post)**2
                pert2_obs_post = c.comm_mem.allreduce(pert2_obs_post_pid)
                variance_obs_post = pert2_obs_post / (c.nens - 1)

            obs_value = obs.obs_seq[obs_rec_id]['obs']
            stats['total_nobs'] += nv * nobs
            stats['omb2'] += np.sum((obs_value - mean_obs_prior)**2)
            stats['varo'] += np.sum(obs.obs_seq[obs_rec_id]['err_std']**2) * nv
            stats['varb'] += np.sum(variance_obs_prior)
            if obs.obs_post_seq:
                stats['amb2'] += np.sum((mean_obs_post - mean_obs_prior)**2)
                stats['omaamb'] += np.sum((obs_value - mean_obs_post)*(mean_obs_post - mean_obs_prior))
                stats['vara'] += np.sum(variance_obs_post)
        return stats

    @abstractmethod
    def adaptive_prior_inflation(self, c, state, obs):
        pass

    @abstractmethod
    def adaptive_post_inflation(self, c, state, obs):
        pass

    @abstractmethod
    def apply_inflation(self, c, state, flag):
        pass
