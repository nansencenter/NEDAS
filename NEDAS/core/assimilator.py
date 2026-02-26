from __future__ import annotations
from typing import TYPE_CHECKING
import os
import inspect
import importlib
from abc import ABC, abstractmethod
import numpy as np
from NEDAS.config import parse_config
from NEDAS.utils.parallel import bcast_by_root
from NEDAS.utils.progress import timer
if TYPE_CHECKING:
    from .context import Context

"""
Assimilator classes
"""

registry = {
    'ETKF': 'ETKFAssimilator',
    'EAKF': 'EAKFAssimilator',
    'TopazDEnKF': 'TopazDEnKFAssimilator',
    #'PDAF': 'PDAFAssimilator',
    #'RHF'
}

class Assimilator(ABC):
    def __init__(self, c: Context):
        #self.analysis_dir = c.io.analysis_dir(c.time, c.iter)

        ##get parameters from config file
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, parse_args=False, **c.config.assimilator_def)
        for key, value in config_dict.items():
            setattr(self, key, value)

    def assimilate(self, c: Context):
        """
        Main method to run the batch assimilation algorithm
        """

        # self.prior_inflation(c)
        # self.partition_grid(c)
        # timer(c)(self.transpose_to_ensemble_complete)(c)
        # timer(c)(self.assimilation_algorithm)(c)
        # self.transpose_to_field_complete(c)
        # self.posterior_inflation(c)

    def prior_inflation(self, c, state, obs):
        """
        Apply covariance inflation for the prior ensemble
        """
        state.output_ens_mean(c, state.fields_prior, state.prior_mean_file)
        c.inflation_func(c, state, obs, 'prior')
        state.output_state(c, state.fields_prior, state.prior_file)

    def posterior_inflation(self, c, state, obs):
        """
        Apply covariance inflation for the posterior ensemble
        """
        obs.prepare_obs_from_state(c, state, 'posterior')  ##update obs_post_seq for stats
        state.output_ens_mean(c, state.fields_post, state.post_mean_file)
        c.inflation_func(c, state, obs, 'posterior')
        state.output_state(c, state.fields_post, state.post_file)

    def partition_grid(self, c, state, obs):
        """
        Partition the analysis grid into several parts and distribute the workload over the mpi ranks.
        """
        state.partitions = bcast_by_root(c.comm)(self.init_partitions)(c)
        obs.obs_inds = bcast_by_root(c.comm_mem)(self.assign_obs)(c, state, obs)
        state.par_list = bcast_by_root(c.comm)(self.distribute_partitions)(c, state, obs)

    def init_partitions(self, c):
        raise NotImplementedError

    def assign_obs(self, c, state, obs):
        raise NotImplementedError

    def distribute_partitions(self, c, state, obs):
        raise NotImplementedError

    def transpose_to_ensemble_complete(self, c, state, obs):
        """
        Communicate among mpi ranks and transpose the locally-stored state/obs chunks to ensemble-complete
        """
        c.print_1p('>>> transpose to ensemble complete:\n')

        c.print_1p('state variables: ')
        state.state_prior = state.transpose_to_ensemble_complete(c, state.fields_prior)

        c.print_1p('z coords: ')
        state.z_state = state.transpose_to_ensemble_complete(c, state.z_fields)

        c.print_1p('obs sequences: ')
        obs.lobs = obs.transpose_to_ensemble_complete(c, state, obs.obs_seq)

        c.print_1p('obs prior sequences: ')
        obs.lobs_prior = obs.transpose_to_ensemble_complete(c, state, obs.obs_prior_seq, ensemble=True)

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


def get_assimilator(c: Context) -> Assimilator:
    """
    Get the correct Assimilator subclass instance based on the configuration.

    Args:
        c (Context): the runtime context object.

    Returns:
        Assimilator: Corresponding Assimilator subclass instance.
    """
    if c.config.assimilator_def is None:
        raise ValueError("assimilator_def not found in Config")
    if 'type' not in c.config.assimilator_def.keys():
        raise KeyError("'type' needs to be specified in assimilator_def")
    assimilator_type = c.config.assimilator_def['type']

    if assimilator_type not in registry:
        raise NotImplementedError(f"Assimilator type '{assimilator_type}' is not implemented")

    module = importlib.import_module('NEDAS.assim_tools.assimilators.'+assimilator_type)
    AssimilatorClass = getattr(module, registry[assimilator_type])

    return AssimilatorClass(c)
