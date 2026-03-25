import os
import inspect
from abc import ABC, abstractmethod
import numpy as np
from NEDAS.config import parse_config
from NEDAS.utils.parallel import bcast_by_root
from .context import Context
from .types import ObsRecordID, PartitionID, ProcIDMem

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
        c.logger('Prior inflation')(c.inflation_func)(c, 'prior')

        # transpose c.state.fields_prior to ensemble-complete c.state.state_prior
        self.partition_grid(c)
        c.logger('Transpose to ensemble-complete')(self.transpose_to_ensemble_complete)(c)

        # the core assimilation algorithm
        # assimilates c.obs.obs_seq into c.state.state_prior to get c.state.state_post
        c.logger('Assimilation algorithm')(self.assimilation_algorithm)(c)

        # transpose c.state.state_post back to field-complete c.state.fields_post
        c.logger('Transpose back to field-complete')(self.transpose_to_field_complete)(c)

        # output the post state
        # TODO: which version of posterior to output? ideally the inflated one? 
        # algorithmically clean way is to output the intermediate versions as well and 
        # let the files be input/output to the inflation func
        # but this is too much IO overhead.
        c.logger('Output posterior ensemble members')(c.state.output_state)(c, 'post')
        c.logger('Output posterior ensemble mean')(c.state.output_ens_mean)(c, 'post')

        if not c.obs.obs_post:
            # for batch filters the obs_post needs to be computed
            # (TODO: they can be updated along with the state, as an alternative)
            c.logger('Prepare obs from posterior state')(c.obs.prepare_obs_from_state)(c, 'post')

        # posterior inflation
        c.logger('Posterior inflation')(c.inflation_func)(c, 'post')

    def partition_grid(self, c: Context) -> None:
        """
        Partition the analysis grid into several parts and distribute the workload over the mpi ranks.
        """
        c.state.partitions = bcast_by_root(c.comm)(self.init_partitions)(c)
        c.obs.obs_inds = bcast_by_root(c.comm_mem)(self.assign_obs)(c)
        c.state.par_list = bcast_by_root(c.comm)(self.distribute_partitions)(c)

    @abstractmethod
    def init_partitions(self, c: Context) -> list:
        """
        Generate spatial partitioning of the domain
        """
        ...

    @abstractmethod
    def assign_obs(self, c: Context) -> dict[ObsRecordID, dict[PartitionID, np.ndarray]]:
        """
        Assign the observation sequence to each partition par_id

        Args:
            c (Context): the runtime context object

        Returns:
            dict[ObsRecordID, dict[PartitionID, np.ndarray]]:
               Indices in the full obs_seq for the subset of obs that belongs to partition par_id
        """
        ...

    @abstractmethod
    def distribute_partitions(self, c: Context) -> dict[ProcIDMem, list[PartitionID]]:
        """
        Distribute partitions across processors
        """
        ...

    def transpose_to_ensemble_complete(self, c: Context) -> None:
        """
        Communicate among mpi ranks and transpose the locally-stored state/obs chunks to ensemble-complete
        """
        c.state.state_prior = c.logger('Transpose prior state')(c.state.transpose_to_ensemble_complete)(c, c.state.fields_prior)

        c.state.state_z = c.logger('Transpose z coordinates')(c.state.transpose_to_ensemble_complete)(c, c.state.fields_z)

        c.obs.lobs = c.logger('Transpose obs sequences')(c.obs.transpose_obs_seq)(c, c.obs.obs_seq)

        c.obs.lobs_prior = c.logger('Transpose obs prior ensemble')(c.obs.transpose_to_ensemble_complete)(c, c.obs.obs_prior)

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
        c.state.fields_post = c.logger('Tranpose posterior state back')(c.state.transpose_to_field_complete)(c, c.state.state_post)

        if c.obs.lobs_post:
            ##TODO there is a bug here, in transpose seq[:, ind] out of bound
            c.obs.obs_post = c.logger('Transpose obs posterior ensemble back')(c.obs.transpose_to_field_complete)(c, c.obs.lobs_post)

    @abstractmethod
    def assimilation_algorithm(self, c: Context) -> None:
        """
        The main assimilation algorithm will be implemented by subclasses
        """
        ...
