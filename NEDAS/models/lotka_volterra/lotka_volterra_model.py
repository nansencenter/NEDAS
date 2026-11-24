import numpy as np
from NEDAS.models.ode_model import OdeModel


class LotkaVolterraModel(OdeModel):
    """
    The competitive Lotka-Volterra (predator-prey) model of 4 species, with the chaotic
    parameters of Vano et al. (2006): dx_i/dt = r_i x_i (1 - sum_j A_ij x_j).

    As in DAPPER mods/LotkaVolterra. The populations are positive, a test for DA of
    bounded variables.

    Args:
        r (list): growth rates
        A (list of lists): interaction coefficients
    """
    r: list
    A: list

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._r = np.asarray(self.r, dtype=float)
        self._A = np.asarray(self.A, dtype=float)
        assert self._A.shape == (len(self._r),) * 2, "A must be a square matrix of the size of r"

    @property
    def state_size(self) -> int:
        return len(self.r)

    def x0(self) -> np.ndarray:
        return np.full(self.state_size, 0.25)

    def dxdt(self, x: np.ndarray) -> np.ndarray:
        return self._r * x * (1. - self._A @ x)
