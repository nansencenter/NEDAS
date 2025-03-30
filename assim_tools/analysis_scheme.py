import os
from utils.shell_utils import makedir
from utils.progress import timer

class AnalysisScheme:
    """
    Class for setting up and running the analysis scheme

    Based on runtime config object, choose the right version of algorithm components:
    state, obs, assimilator, updator, covariance, localization and inflation.

    Run the key steps in the scheme:
    state.prepare_obs, obs.prepare_obs, obs.prepare_obs_from_state, assimilator.assimilate, updator.update
    """
    def validate_mpi_environment(self, c):
        nproc = c.nproc
        nproc_actual = c.comm.Get_size()
        if nproc != nproc_actual:
            raise RuntimeError(f"Error: nproc {nproc} != mpi size {nproc_actual}")

    def init_analysis_dir(self, c):
        self._analysis_dir = c.analysis_dir(c.time, c.scale_id)
        if c.pid == 0:
            makedir(self._analysis_dir)
            print(f"\nRunning assimilation step in {self._analysis_dir}\n", flush=True)

    def get_state(self, c):
        from .state import State
        return State(c)

    def get_obs(self, c, state):
        from .obs import Obs
        return Obs(c, state)

    # def get_covariance(c):
    #     from .covariance.ensemble import EnsembleCovariance as Covariance
    #     return Covariance()

    def get_assimilator(self, c):
        if c.assim_mode == 'batch':
            if c.filter_type == 'TopazDEnKF':
                from .assimilators.TopazDEnKF import TopazDEnKFAssimilator as Assimilator
            elif c.filter_type == 'ETKF':
                from .assimilators.ETKF import ETKFAssimilator as Assimilator
            else:
                raise ValueError(f"Unknown filter_type {c.filter_type} for batch assimilation")

        elif c.assim_mode == 'serial':
            if c.filter_type == 'EAKF':
                from .assimilators.EAKF import EAKFAssimilator as Assimilator
    #        elif c.filter_type == 'RHF':
    #            from .assimilators.RHF import RHFAssimilator as Assimilator
    #        elif c.filter_type == 'QCEKF':
    #            from .assimilators.QCEKF import QCEKFAssimilator as Assimilator
            else:
                raise ValueError(f"Unknown filter_type {c.filter_type} for serial assimilation")

        else:
            raise ValueError(f"Unknown assim_mode {c.assim_mode}")

        return Assimilator(c)

    def get_updator(self, c):
        if c.run_alignment and c.scale_id < c.nscale-1:
            if c.interp_displace:
                from .updators.alignment_interp import AlignmentUpdator as Updator
            else:
                from .updators.alignment import AlignmentUpdator as Updator
        else:
            from .updators.additive import AdditiveUpdator as Updator
        return Updator(c)

    def get_localization_funcs(self, c):
        local_funcs = {}
        for key in ['horizontal', 'vertical', 'temporal']:
            if c.localization[key]:
                local_funcs[key] = self.get_localization_func_component(c.localization[key])
            else:
                local_funcs[key]
        return local_funcs

    def get_localization_func_component(self, localization_type):
        localization_types = localization_type.split(',')

        ##distance-based localization schemes
        if 'GC' in localization_types:
            from .localization.distance_based import local_func_GC as local_func
        elif 'step' in localization_types:
            from .localization.distance_based import local_func_step as local_func
        elif 'exp' in localization_types:
            from .localization.distance_based import local_func_exp as local_func

        # ##correlation based localization schemes
        # elif 'SER' in localization_types:
        #     from .localization.SER import CorrelationBasedLocalization as Localization
        # else:
        #     raise ValueError(f"Unknown localization type {type}")
        return local_func

    def get_inflation_func(self, c):
        if c.inflation:
            inflation_type = c.inflation.get('type', '').split(',')
            if 'multiplicative' in inflation_type:
                from .inflation.multiplicative import MultiplicativeInflation as Inflation
            elif 'RTPP' in inflation_type:
                from .inflation.RTPP import RTPPInflation as Inflation
            else:
                from .inflation.base import Inflation
        else:
            from .inflation.base import Inflation
        return Inflation(c)

    def get_misc_transform(self, c):
        if c.nscale > 1:
            from .misc_transform.scale_bandpass import ScaleBandpassTransform as Transform
        else:
            from .misc_transform.identity import Transform
        return Transform(c)

    def __call__(self, c):
        """
        Main method for performing the analysis step
        Input:
        - c: config object obtained at runtime
        """
        self.validate_mpi_environment(c)

        ##multiscale approach: loop over scale components and perform assimilation on each scale
        ##more complex outer loops can be implemented here
        analysis_grid = c.grid
        for c.scale_id in range(c.nscale):
            self.init_analysis_dir(c)
            c.grid = analysis_grid.change_resolution_level(c.resolution_level[c.scale_id])
            c.misc_transform = self.get_misc_transform(c)
            c.localization_funcs = self.get_localization_funcs(c)
            c.inflation_func = self.get_inflation_func(c)

            state = self.get_state(c)
            timer(c)(state.prepare_state)(c)

            obs = self.get_obs(c, state)
            timer(c)(obs.prepare_obs)(c, state)
            timer(c)(obs.prepare_obs_from_state)(c, state, 'prior')

            assimilator = self.get_assimilator(c)
            timer(c)(assimilator.assimilate)(c, state, obs)

            updator = self.get_updator(c)
            timer(c)(updator.update)(c, state)
