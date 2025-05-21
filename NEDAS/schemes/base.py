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

class AnalysisScheme(ABC):
    """
    Base class for setting up and running the analysis scheme.

    Based on runtime config object, choose the right version of algorithm components:
    state, obs, assimilator, updator, covariance, localization and inflation.

    Run the key steps in the scheme:
    state.prepare_obs, obs.prepare_obs, obs.prepare_obs_from_state, assimilator.assimilate, updator.update
    """
    @abstractmethod
    def run(self, c: Type['Config']) -> None:
        """
        Main workflow to run the analysis scheme.

        Args:
            c (Config): Configuration object.
        """
        pass

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
        self._analysis_dir = c.analysis_dir(c.time, c.step)
        if c.pid == 0:
            makedir(self._analysis_dir)
            print(f"\nRunning assimilation step in {self._analysis_dir}\n", flush=True)
