import numpy as np
from scipy.ndimage import convolve1d
from NEDAS.models.ode_model import OdeModel


def _sum_weights(width: int) -> tuple[np.ndarray, np.ndarray]:
    """weights and offsets of the modified sum of Lorenz (2005), the boxcar of `width` points,
    with half weights at both ends when width is even"""
    r = width // 2
    w = np.ones(2 * r + 1)
    if width != len(w):
        w[0] = w[-1] = 0.5
    return w, np.arange(-r, r + 1)


class Lorenz05Model(OdeModel):
    """
    Lorenz (2005) Models II and III: Lorenz-96 with spatially smooth large-scale waves (Model II),
    plus superimposed small-scale activity coupled to them (Model III).

    As in DART models/lorenz_04 (coded in 2004 by J. Hansen) and DAPPER mods/Lorenz05, whose
    vectorized formulation is used here. I=1 gives Model II, and also K=1 gives Lorenz-96.

    Args:
        nx (int): state vector length
        K (int): width of the smoothing kernel in the advection term (larger = smoother, longer waves)
        I (int): half-width of the filter separating the large and small scales
        space_time_scale (float): scaling of the small-scale variability (b in the paper)
        coupling (float): coupling strength of the scales (c in the paper)
        F (float): forcing
    """
    cyclic = True
    nx: int
    K: int
    I: int
    space_time_scale: float
    coupling: float
    F: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        I = self.I
        alpha = (3 * I**2 + 3) / (2 * I**3 + 4 * I)
        beta = (2 * I**2 + 1) / (I**4 + 2 * I**2)
        w, offsets = _sum_weights(2 * I)
        self._filter = w * (alpha - beta * np.abs(offsets))
        self._boxcar = _sum_weights(self.K)[0] / self.K

    @property
    def state_size(self) -> int:
        return self.nx

    def x0(self) -> np.ndarray:
        x = np.full(self.nx, 7.)
        x[0] += 1.  # off the fixed point
        return x

    def decompose(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """the large-scale (x) and small-scale (y) parts of the state"""
        x = convolve1d(z, self._filter, mode='wrap') if self.I > 1 else z
        return x, z - x

    def _prodsum_self(self, x: np.ndarray) -> np.ndarray:
        """[x, x]_K of Lorenz (2005), eqn 10"""
        K = self.K
        W = convolve1d(x, self._boxcar, mode='wrap')
        return -np.roll(W, 2 * K) * np.roll(W, K) + convolve1d(np.roll(W, K) * np.roll(x, -K), self._boxcar, mode='wrap')

    @staticmethod
    def _prodsum_1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """[x, y]_1"""
        return -np.roll(x, 2) * np.roll(y, 1) + np.roll(x, 1) * np.roll(y, -1)

    def dxdt(self, z: np.ndarray) -> np.ndarray:
        x, y = self.decompose(z)
        dz = self._prodsum_self(x) - x + self.F
        if self.I > 1:
            b, c = self.space_time_scale, self.coupling
            dz += self._prodsum_1(y, y) * b**2 + self._prodsum_1(y, x) * c - y * b
        return dz
