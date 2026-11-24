import os
import inspect
from abc import ABC, abstractmethod
import numpy as np
from NEDAS.config import parse_config
from NEDAS.utils.parallel import bcast_by_root
from .context import Context
from .types import ObsRecordID, PartitionID, ProcIDMem

class Assimilator(ABC):
    assim_mode: str

    # capabilities of the algorithm, subclasses set the ones they support to True;
    # check_capabilities() matches them against the settings of the other components
    supports_static_members: bool = False       # covariance_def.nens_static > 0
    supports_hybrid_perturbation: bool = False  # covariance_def.hybrid_perturbation

    def __init__(self, c: Context):
        # get parameters from config file
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, parse_args=False, **c.config.assimilator_def)
        for key, value in config_dict.items():
            setattr(self, key, value)

    def check_capabilities(self, c: Context) -> None:
        """
        Check that the algorithm supports the settings of the other assimilation components,
        an unsupported one would otherwise be silently ignored by the algorithm
        """
        unsupported = []
        if c.covariance.nens_static > 0 and not self.supports_static_members:
            unsupported.append('covariance_def.nens_static > 0')
        if c.covariance.hybrid_perturbation and c.covariance.beta > 0 and not self.supports_hybrid_perturbation:
            unsupported.append('covariance_def.hybrid_perturbation')
        if unsupported:
            raise NotImplementedError(f"{self.__class__.__name__} does not support: {', '.join(unsupported)}")

    def assimilate(self, c: Context):
        """
        Main method to run the batch assimilation algorithm
        """
        # prior inflation step: 'once_after_outer_loop'-style timing (see
        # core/inflation.py::Inflation.timing) skips this per-iteration application --
        # schemes/filter.py::filter() applies it once, before the outer loop, via
        # apply_inflation_once() instead (found and fixed 2026-07-28, Yue: this call was
        # previously unconditional, so prior inflation had no "once" mode at all regardless of
        # the configured timing -- only posterior inflation's per-iteration call already had
        # this gate).
        if c.inflation_func.timing == 'per_iteration':
            c.logger('Prior inflation')(c.inflation_func)(c, 'prior')

        # transpose c.state.fields_prior to ensemble-complete c.state.state_prior
        self.partition_grid(c)
        c.logger('Transpose to ensemble-complete')(self.transpose_to_ensemble_complete)(c)

        # the core assimilation algorithm
        # assimilates c.obs.obs_seq into c.state.state_prior to get c.state.state_post
        c.logger('Assimilation algorithm')(self.assimilation_algorithm)(c)

        # reduce scalar parameter updates across MPI ranks and write posteriors to models
        if c.state.info.scalars:
            c.logger('Reduce scalar parameters')(c.state.reduce_scalars)(c)
            c.logger('Output posterior scalar parameters')(c.state.output_scalar_variables)(c, 'post')

        # transpose c.state.state_post back to field-complete c.state.fields_post
        c.logger('Transpose back to field-complete')(self.transpose_to_field_complete)(c)

        # batch assimilators don't populate obs_post internally (unlike serial, which builds it
        # up incrementally during the local obs loop -- see assim_tools/assimilators/serial.py),
        # so recompute it here from the just-transposed fields_post before posterior inflation
        # needs it. schemes/filter.py's filter_iter recomputes
        # obs_post again afterward (for batch mode) to reflect the final, post-update/post-inflation
        # state, so skipping this call otherwise avoids a redundant forward-operator pass.
        if (self.assim_mode == 'batch' and c.inflation_func.post and c.inflation_func.adaptive
                and c.inflation_func.timing == 'per_iteration'):
            c.logger('Prepare obs from post state (for posterior inflation)')(c.obs.prepare_obs_from_state)(c, 'post')

        # output the post state
        # TODO: which version of posterior to output? ideally the inflated one?
        # algorithmically clean way is to output the intermediate versions as well and
        # let the files be input/output to the inflation func
        # but this is too much IO overhead.
        c.logger('Output posterior ensemble members')(c.state.output_state)(c, 'post')
        c.logger('Output posterior ensemble mean')(c.state.output_ens_mean)(c, 'post')

        # posterior inflation: 'once_after_outer_loop' skips this per-iteration application --
        # schemes/filter.py::final_inflation applies it once, after the outer loop, on the full
        # recombined state instead (see core/inflation.py's Inflation.timing docstring)
        if c.inflation_func.timing == 'per_iteration':
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
        c.state.state_prior = c.logger('Transpose prior state')(c.state.transpose_to_ensemble_complete)(c, c.state.fields_prior, c.mem_list)

        c.state.state_z = c.logger('Transpose z coordinates')(c.state.transpose_to_ensemble_complete)(c, c.state.fields_z, c.mem_list)

        c.obs.lobs = c.logger('Transpose obs sequences')(c.obs.transpose_obs_seq)(c, c.obs.obs_seq)

        c.obs.lobs_prior = c.logger('Transpose obs prior ensemble')(c.obs.transpose_to_ensemble_complete)(c, c.obs.obs_prior, c.mem_list)

        # static members (covariance_def.nens_static), a separate batch with its own mem_list
        if c.nens_static > 0:
            c.state.state_static = c.logger('Transpose static state')(c.state.transpose_to_ensemble_complete)(c, c.state.fields_static, c.mem_list_static)
            c.obs.lobs_prior_static = c.logger('Transpose static obs prior ensemble')(c.obs.transpose_to_ensemble_complete)(c, c.obs.obs_prior_static, c.mem_list_static)

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
            c.obs.obs_post = c.logger('Transpose obs posterior ensemble back')(c.obs.transpose_to_field_complete)(c, c.obs.lobs_post)

    @abstractmethod
    def assimilation_algorithm(self, c: Context) -> None:
        """
        The main assimilation algorithm will be implemented by subclasses
        """
        ...
