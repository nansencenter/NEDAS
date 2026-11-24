import numpy as np
from NEDAS.models.ode_model import OdeModel


class Lorenz63Model(OdeModel):
    """
    The Lorenz (1963) convection model, 3 variables.

    As in DART models/lorenz_63 and DAPPER mods/Lorenz63. The default time step and the
    6-hourly cycle of 0.25 time units follow the Sakov et al. (2012) setup in DAPPER.

    Args:
        sigma, rho, beta (float): the parameters, by default the classic chaotic ones
    """
    sigma: float
    rho: float
    beta: float

    @property
    def state_size(self) -> int:
        return 3

    def x0(self) -> np.ndarray:
        return np.array([1.509, -1.531, 25.46])

    def dxdt(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.sigma * (x[1] - x[0]),
                         self.rho * x[0] - x[1] - x[0] * x[2],
                         x[0] * x[1] - self.beta * x[2]])
