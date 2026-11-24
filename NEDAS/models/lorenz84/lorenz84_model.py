import numpy as np
from NEDAS.models.ode_model import OdeModel


class Lorenz84Model(OdeModel):
    """
    The Lorenz (1984) model of the general circulation, 3 variables: the strength of the westerly
    wind (x) and the cosine and sine phases of a superposed large-scale eddy (y, z).

    As in DART models/lorenz_84 and DAPPER mods/Lorenz84.

    Args:
        a, b (float): damping and displacement of the eddies by the westerly current
        F, G (float): the thermal forcings of the westerly current and of the eddies
    """
    a: float
    b: float
    F: float
    G: float

    @property
    def state_size(self) -> int:
        return 3

    def x0(self) -> np.ndarray:
        return np.array([1.65, 0.49, 1.21])

    def dxdt(self, x: np.ndarray) -> np.ndarray:
        x, y, z = x
        return np.array([-y**2 - z**2 - self.a * x + self.a * self.F,
                         x * y - self.b * x * z - y + self.G,
                         self.b * x * y + x * z - z])
