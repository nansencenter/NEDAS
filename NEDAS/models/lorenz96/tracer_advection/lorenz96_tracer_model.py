import numpy as np
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.netcdf_lib import nc_write_var
from .core import M_nl
from ..lorenz96_model import Lorenz96Model


class Lorenz96TracerModel(Lorenz96Model):
    """
    Lorenz 1996 model extended with a tracer advection component.

    State vector is [lorenz_state (nx), tracer (nx), source (nx)].
    """
    mean_velocity: float
    pert_velocity_multiplier: float
    diffusion_coef: float
    e_folding: float
    sink_rate: float
    point_tracer_source_rate: float
    positive_tracer: bool
    bound_above_is_one: bool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # state stores lorenz + tracer + source components as a flat 3*nx array
        self._state_size = 3 * self.nx

    def write_var_to_file(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        var_name = self.variables[name].name
        assert isinstance(var_name, str)
        nc_write_var(fname, {'t': None, 'x': self._state_size}, var_name, var, recno={'t': 0})

    def generate_initial_condition(self):
        state = np.zeros(self._state_size)
        state[0:self.nx] = np.random.normal(0, 1, self.nx)
        if self.positive_tracer:
            state[2*self.nx] = self.point_tracer_source_rate
        else:
            state[2*self.nx] = -self.point_tracer_source_rate
        return state

    def run(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        self.run_status = 'running'

        state = self.read_var(**kwargs)
        next_time = kwargs['time'] + kwargs['forecast_period'] * dt1h
        next_state = M_nl(state, kwargs['forecast_period'] / self.hours_per_unit_time,
                          self.F, self.dt,
                          self.mean_velocity, self.pert_velocity_multiplier, self.diffusion_coef,
                          self.e_folding, self.sink_rate, self.bound_above_is_one, self.positive_tracer)
        self.write_var(next_state, **{**kwargs, 'time': next_time})

        self.run_status = 'complete'
