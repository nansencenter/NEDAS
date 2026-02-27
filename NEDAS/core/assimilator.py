import os
import inspect
from abc import ABC, abstractmethod
import numpy as np
from NEDAS.config import parse_config
from NEDAS.utils.parallel import bcast_by_root
from NEDAS.utils.progress import timer
from .context import Context

class Assimilator(ABC):
    def __init__(self, c: Context):
        ##get parameters from config file
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, parse_args=False, **c.config.assimilator_def)
        for key, value in config_dict.items():
            setattr(self, key, value)

    def assimilate(self, c: Context):
        """
        Main method to run the batch assimilation algorithm
        """
        # prior inflation step
        c.inflation_func(c, 'prior')

        # transpose c.state.fields_prior to ensemble-complete c.state.state_prior
        self.partition_grid(c)
        timer(c)(self.transpose_to_ensemble_complete)(c)

        # the core assimilation algorithm
        # assimilates c.obs.obs_seq into c.state.state_prior to get c.state.state_post
        timer(c)(self.assimilation_algorithm)(c)

        # transpose c.state.state_post back to field-complete c.state.fields_post
        self.transpose_to_field_complete(c)

        # posterior inflation
        c.inflation_func(c, 'posterior')

    # def prior_inflation(self, c, state, obs):
    #     """
    #     Apply covariance inflation for the prior ensemble
    #     """
    #     state.output_ens_mean(c, state.fields_prior, state.prior_mean_file)
    #     c.inflation_func(c, state, obs, 'prior')
    #     state.output_state(c, state.fields_prior, state.prior_file)

    # def posterior_inflation(self, c, state, obs):
    #     """
    #     Apply covariance inflation for the posterior ensemble
    #     """
    #     obs.prepare_obs_from_state(c, state, 'posterior')  ##update obs_post_seq for stats
    #     state.output_ens_mean(c, state.fields_post, state.post_mean_file)
    #     c.inflation_func(c, state, obs, 'posterior')
    #     state.output_state(c, state.fields_post, state.post_file)

    def partition_grid(self, c: Context) -> None:
        """
        Partition the analysis grid into several parts and distribute the workload over the mpi ranks.
        """
        c.state.partitions = bcast_by_root(c.comm)(self.init_partitions)(c)
        c.obs.obs_inds = bcast_by_root(c.comm_mem)(self.assign_obs)(c)
        c.state.par_list = bcast_by_root(c.comm)(self.distribute_partitions)(c)

    @abstractmethod
    def init_partitions(self, c: Context) -> list[tuple]:
        ...

    @abstractmethod
    def assign_obs(self, c: Context) -> dict:
        ...

    @abstractmethod
    def distribute_partitions(self, c: Context) -> dict[int, list[int]]:
        ...

    def transpose_to_ensemble_complete(self, c: Context) -> None:
        """
        Communicate among mpi ranks and transpose the locally-stored state/obs chunks to ensemble-complete
        """
        c.print_1p('>>> transpose to ensemble complete:\n')

        c.print_1p('state variables: ')
        c.state.state_prior = c.state.transpose_to_ensemble_complete(c, c.state.fields_prior)

        c.print_1p('z coords: ')
        c.state.z_state = c.state.transpose_to_ensemble_complete(c, c.state.z_fields)

        c.print_1p('obs sequences: ')
        c.obs.lobs = c.obs.transpose_obs_seq(c, c.obs.obs_seq)

        c.print_1p('obs prior sequences: ')
        c.obs.lobs_prior = c.obs.transpose_to_ensemble_complete(c, c.obs.obs_prior)

        # if c.debug:
        #     np.save(os.path.join(self.analysis_dir, f'state_prior.{c.pid_mem}.{c.pid_rec}.npy'), state.state_prior)
        #     np.save(os.path.join(self.analysis_dir, f'z_state.{c.pid_mem}.{c.pid_rec}.npy'), state.z_state)
        #     np.save(os.path.join(self.analysis_dir, f'lobs.{c.pid_mem}.{c.pid_rec}.npy'), obs.lobs)
        #     np.save(os.path.join(self.analysis_dir, f'lobs_prior.{c.pid_mem}.{c.pid_rec}.npy'), obs.lobs_prior)

    def transpose_to_field_complete(self, c: Context):
        """
        Communicate among mpi ranks and transpose the locally-stored state/obs chunks
        back to field-complete
        """
        c.print_1p('>>> transpose back to field complete\n')

        c.print_1p('state variables: ')
        c.state.fields_post = c.state.transpose_to_field_complete(c, c.state.state_post)

        # c.print_1p('obs prior sequences: ')
        ##TODO there is a bug here, in transpose seq[:, ind] out of bound
        # obs.obs_post_seq = obs.transpose_to_field_complete(c, state, obs.lobs_post)

    @abstractmethod
    def assimilation_algorithm(self, c: Context) -> None:
        """
        The main assimilation algorithm will be implemented by subclasses
        """
        ...
