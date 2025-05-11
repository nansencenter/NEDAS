"""
The schemes module contain workflows known as analysis schemes.

Functions:
    :func:`get_analysis_scheme`: Factory function to get the right AnalysisScheme subclass based on configuration.

Classes:
    :class:`AnalysisScheme`:
        Base class for setting up and running the analysis scheme.

    :class:`OfflineFilterAnalysisScheme`:
        Subclass for offline filter analysis scheme.
        Running the filter at each time step (analysis cycle), and run the ensemble forecast
        forward in time, reaching the next cycle, then repeat the process.
"""
from abc import ABC, abstractmethod
from typing import Type, TYPE_CHECKING
from NEDAS.utils.shell_utils import makedir
if TYPE_CHECKING:
    from NEDAS.config import Config
from NEDAS.assim_tools.state import get_state_class
from NEDAS.assim_tools.obs import get_obs_class

class AnalysisScheme(ABC):
    """
    Base class for setting up and running the analysis scheme.

    Based on runtime config object, choose the right version of algorithm components:
    state, obs, assimilator, updator, covariance, localization and inflation.

    Run the key steps in the scheme:
    state.prepare_obs, obs.prepare_obs, obs.prepare_obs_from_state, assimilator.assimilate, updator.update
    """
    def validate_mpi_environment(self, c: Type['Config']):
        """
        Validate the MPI environment and ensure the number of processes is consistent.
        """
        nproc = c.nproc
        nproc_actual = c.comm.Get_size()
        if nproc != nproc_actual:
            raise RuntimeError(f"Error: nproc {nproc} != mpi size {nproc_actual}")

    def init_analysis_dir(self, c) -> None:
        """
        Initialize the analysis directory.
        """
        self._analysis_dir = c.analysis_dir(c.time, c.scale_id)
        if c.pid == 0:
            makedir(self._analysis_dir)
            print(f"\nRunning assimilation step in {self._analysis_dir}\n", flush=True)

    # def get_covariance(c):
    #     from NEDAS.assim_tools.covariance.ensemble import EnsembleCovariance as Covariance
    #     return Covariance()

    def get_updator(self, c):
        if c.alignment and c.scale_id < c.nscale-1:
            if c.alignment['interp_displace']:
                from NEDAS.assim_tools.updators.alignment_interp import AlignmentUpdator as Updator
            else:
                from NEDAS.assim_tools.updators.alignment import AlignmentUpdator as Updator
        else:
            from NEDAS.assim_tools.updators.additive import AdditiveUpdator as Updator
        return Updator(c)

    def get_localization_funcs(self, c):
        local_funcs = {}
        for key in ['horizontal', 'vertical', 'temporal']:
            if c.localization[key]:
                local_funcs[key] = self.get_localization_func_component(c.localization[key])
            else:
                local_funcs[key] = None
        return local_funcs

    def get_localization_func_component(self, localization_type):
        localization_types = localization_type.split(',')

        ##distance-based localization schemes
        if 'GC' in localization_types:
            from NEDAS.assim_tools.localization.distance_based import local_func_GC as local_func
        elif 'step' in localization_types:
            from NEDAS.assim_tools.localization.distance_based import local_func_step as local_func
        elif 'exp' in localization_types:
            from NEDAS.assim_tools.localization.distance_based import local_func_exp as local_func

        # ##correlation based localization schemes
        # elif 'SER' in localization_types:
        #     from NEDAS.assim_tools.localization.SER import CorrelationBasedLocalization as Localization
        # elif 'NICE' in localization_types:
        #     from NEDAS.assim_tools.localization.NICE import 
        # else:
        #     raise ValueError(f"Unknown localization type {type}")
        return local_func

    def get_inflation_func(self, c):
        if c.inflation:
            inflation_type = c.inflation.get('type', '').split(',')
            if 'multiplicative' in inflation_type:
                from NEDAS.assim_tools.inflation.multiplicative import MultiplicativeInflation as Inflation
            elif 'RTPP' in inflation_type:
                from NEDAS.assim_tools.inflation.RTPP import RTPPInflation as Inflation
            else:
                from NEDAS.assim_tools.inflation.base import Inflation
        else:
            from NEDAS.assim_tools.inflation.base import Inflation
        return Inflation(c)

    def get_misc_transform(self, c):
        if c.nscale > 1:
            from NEDAS.assim_tools.misc_transform.scale_bandpass import ScaleBandpassTransform as Transform
        else:
            from NEDAS.assim_tools.misc_transform.identity import Transform
        return Transform(c)

    @abstractmethod
    def run(self) -> None:
        """
        Main function to run the analysis scheme.
        """
        pass

