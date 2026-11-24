import numpy as np
from NEDAS.models.ode_model import OdeModel


class IkedaModel(OdeModel):
    """
    The Ikeda map, a discrete-time chaotic system of 2 variables, with a strongly curved,
    non-Gaussian attractor.

    As in DART models/ikeda and DAPPER mods/Ikeda. A time step (dt = 1) is one iteration.

    Args:
        u (float): the parameter of the map, chaotic for u > 0.6
    """
    u: float

    @property
    def state_size(self) -> int:
        return 2

    def x0(self) -> np.ndarray:
        return np.zeros(2)

    def step(self, x: np.ndarray) -> np.ndarray:
        t = 0.4 - 6. / (1. + x[0]**2 + x[1]**2)
        return np.array([1. + self.u * (x[0] * np.cos(t) - x[1] * np.sin(t)),
                         self.u * (x[0] * np.sin(t) + x[1] * np.cos(t))])
