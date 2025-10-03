import os
import numpy as np
from NEDAS.grid import Grid1D
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.shell_utils import run_command, makedir
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from .core import M_nl
from ..lorenz96_model import Lorenz96Model

class Lorenz96TracerModel(Lorenz96Model):
    nx: int
    F: float
    dt: float
    restart_dt: float
    mean_velocity: float
    pert_velocity_multiplier: float
    diffusion_coef: float
    e_folding: float
    sink_rate: float
    source_rate: float
    point_tracer_source_rate: float
    positive_tracer: bool
    bound_above_is_one: bool

    def generate_initial_condition(self):
        state = np.zeros(3*self.nx)
        state[0:self.nx] = np.random.normal(0, 1, self.nx)
        # state[0] = 0.1
        if self.positive_tracer:
            state[2*self.nx] = self.point_tracer_source_rate
        else:
            state[2*self.nx] = -self.point_tracer_source_rate
        return state

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'
        state = self.read_var(**kwargs)
        makedir(kwargs['path'])

        next_time = kwargs['time'] + kwargs['forecast_period'] * dt1h
        next_state = M_nl(state, kwargs['forecast_period']/24, self.F, self.dt,
                          self.mean_velocity, self.pert_velocity_multiplier, self.diffusion_coef,
                          self.e_folding, self.sink_rate, self.bound_above_is_one, self.positive_tracer)
        self.write_var(next_state, **{**kwargs, 'time':next_time})
        self.run_status = 'complete'
