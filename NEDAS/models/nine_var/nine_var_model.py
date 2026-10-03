import numpy as np
from NEDAS.models.ode_model import OdeModel


class NineVarModel(OdeModel):
    """
    The Lorenz (1980) nine-variable primitive equation model: three Fourier modes of the
    velocity potential (x), streamfunction (y) and geopotential height (z) on a doubly periodic
    domain, with forcing in the first mode. Its attractor has slow (balanced) and fast (gravity
    wave) motions, a test of how DA handles balance.

    As in DART models/9var, with its time step and two-stage scheme (12 steps per time unit,
    an hour each). The state vector is x1, x2, x3, y1, y2, y3, z1, z2, z3.

    Args:
        g (float): the gravity parameter; 8 is Lorenz's, 9.90 gives a higher dimensional attractor
    """
    g: float
    a = np.array([1., 1., 3.])
    b = np.array([-1.5, -1.5, 0.5])
    f = np.array([0.1, 0., 0.])
    h0 = np.array([-1., 0., 0.])
    nu = 1. / 48.
    kappa = 1. / 48.
    c_ = 0.8660254

    @property
    def state_size(self) -> int:
        return 9

    def x0(self) -> np.ndarray:
        return np.full(9, 0.1)

    def dxdt(self, s: np.ndarray) -> np.ndarray:
        a, b, f, h, nu, kappa, c, g = self.a, self.b, self.f, self.h0, self.nu, self.kappa, self.c_, self.g
        x, y, z = s[0:3], s[3:6], s[6:9]
        j = [1, 2, 0]  # J = mod(i, 3) + 1 for i = 1, 2, 3, as 0-based indices
        k = [2, 0, 1]  # K = mod(i + 1, 3) + 1
        xj, xk, yj, yk, zj, zk = x[j], x[k], y[j], y[k], z[j], z[k]
        aj, ak, bj, bk, hj, hk = a[j], a[k], b[j], b[k], h[j], h[k]
        dx = (a * b * xj * xk - c * (a - ak) * xj * yk + c * (a - aj) * yj * xk - 2 * c**2 * yj * yk
              - nu * a**2 * x + a * y - a * z) / a
        dy = (-ak * bk * xj * yk - aj * bj * yj * xk + c * (ak - aj) * yj * yk - a * x - nu * a**2 * y) / a
        dz = (-bk * xj * (zk - hk) - bj * (zj - hj) * xk + c * yj * (zk - hk) - c * (zj - hj) * yk
              + g * a * x - kappa * a * z + f)
        return np.concatenate([dx, dy, dz])

    def step(self, x: np.ndarray) -> np.ndarray:
        """DART's two-stage scheme"""
        x1 = x + self.dt * self.dxdt(x)
        x2 = x1 + self.dt * self.dxdt(x1)
        return 0.5 * (x + x2)
