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
                 prior: bool=False, post: bool=False):
        self.coef = coef
        self.adaptive = adaptive
        self.prior = prior
        self.post = post

    def __call__(self, c: Context, flag: Literal['prior', 'post']) -> None:
        """
        Perform the covariance inflation method
        """
        if flag == 'prior' and self.prior:
            if self.adaptive:
                self.adaptive_prior_inflation(c)
            self.apply_inflation(c, flag)

        if flag == 'post' and self.post:
            if self.adaptive:
                self.adaptive_post_inflation(c)
            self.apply_inflation(c, flag)
        # c.state.output_state(c, flag)
        # c.state.output_ens_mean(c, flag)

    def obs_space_stats(self, c: Context):
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
        for r, obs_rec_id in enumerate(c.obs.obs_rec_list[c.pid_rec]):
            obs_rec = c.obs.info.records[obs_rec_id]
            nobs = obs_rec.nobs

            ##1. get ensemble mean obs_prior:
            if obs_rec.is_vector:
                nv = 2
                shape = (nv, nobs)
            else:
                nv = 1
                shape = (nobs,)

            ##sum over all obs_prior_seq locally stored on pid
            sum_obs_prior_pid = np.zeros(shape)
            for mem_id in c.mem_list[c.pid_mem]:
                sum_obs_prior_pid += c.obs.obs_prior[mem_id, obs_rec_id]
            ##sum over all obs_prior_seq on differnet pids to get the total sum
            sum_obs_prior = c.comm_mem.allreduce(sum_obs_prior_pid)
            mean_obs_prior = sum_obs_prior / c.config.nens
            mean_obs_post = None

            if c.obs.obs_post:
                ##sum over all obs_prior_seq locally stored on pid
                sum_obs_post_pid = np.zeros(shape)
                for mem_id in c.mem_list[c.pid_mem]:
                    sum_obs_post_pid += c.obs.obs_post[mem_id, obs_rec_id]
                ##sum over all obs_prior_seq on differnet pids to get the total sum
                sum_obs_post = c.comm_mem.allreduce(sum_obs_post_pid)
                mean_obs_post = sum_obs_post / c.config.nens

            ##2. get ensemble spread obs_prior:
            pert2_obs_prior_pid = np.zeros(shape)
            for mem_id in c.mem_list[c.pid_mem]:
                pert2_obs_prior_pid += (c.obs.obs_prior[mem_id, obs_rec_id] - mean_obs_prior)**2
            pert2_obs_prior = c.comm_mem.allreduce(pert2_obs_prior_pid)
            variance_obs_prior = pert2_obs_prior / (c.config.nens - 1)
            variance_obs_post = None

            if c.obs.obs_post:
                pert2_obs_post_pid = np.zeros(shape)
                for mem_id in c.mem_list[c.pid_mem]:
                    pert2_obs_post_pid += (c.obs.obs_post[mem_id, obs_rec_id] - mean_obs_post)**2
                pert2_obs_post = c.comm_mem.allreduce(pert2_obs_post_pid)
                variance_obs_post = pert2_obs_post / (c.config.nens - 1)

            obs_value = c.obs.obs_seq[obs_rec_id]['obs']
            stats['total_nobs'] += nv * nobs
            stats['omb2'] += np.sum((obs_value - mean_obs_prior)**2)
            stats['varo'] += np.sum(c.obs.obs_seq[obs_rec_id]['err_std']**2) * nv
            stats['varb'] += np.sum(variance_obs_prior)
            if c.obs.obs_post and variance_obs_post is not None:
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
