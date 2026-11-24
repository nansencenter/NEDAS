import numpy as np
from NEDAS.models.ode_model import OdeModel


class KSModel(OdeModel):
    """
    The Kuramoto-Sivashinsky equation u_t = -u u_x - u_xx - u_xxxx, on a periodic domain of length
    32 pi (by default), the simplest PDE with spatio-temporal chaos.

    As in DAPPER mods/KS: pseudo-spectral, with the ETD-RK4 scheme of Kassam and Trefethen (2005).
    The grid is the nx collocation points, so distances are in grid spacings (L / nx).

    Args:
        nx (int): number of grid points
        DL (float): domain length in units of pi
    """
    cyclic = True
    nx: int
    DL: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        h = self.dt
        kk = np.append(np.arange(0, self.nx // 2), 0) * 2. / self.DL  # rfft wavenumbers, no Nyquist
        L = kk**2 - kk**4  # the linear operator, F[-u_xx - u_xxxx]
        self._D = 1j * kk
        self._E = np.exp(h * L)
        self._E2 = np.exp(h * L / 2)
        roots = np.exp(1j * np.pi * (0.5 + np.arange(16)) / 16)
        CL = h * L[:, None] + roots  # contour integrals avoid the cancellation at small L
        self._Q = h * ((np.exp(CL / 2) - 1) / CL).mean(axis=-1).real
        self._f1 = h * ((-4 - CL + np.exp(CL) * (4 - 3 * CL + CL**2)) / CL**3).mean(axis=-1).real
        self._f2 = h * ((2 + CL + np.exp(CL) * (-2 + CL)) / CL**3).mean(axis=-1).real
        self._f3 = h * ((-4 - 3 * CL - CL**2 + np.exp(CL) * (4 - CL)) / CL**3).mean(axis=-1).real

    @property
    def state_size(self) -> int:
        return self.nx

    def x0(self) -> np.ndarray:
        """the initial condition of Kassam and Trefethen (2005)"""
        x = self.DL * np.pi * np.linspace(0, 1, self.nx + 1)[1:]
        return np.cos(x / 16) * (1 + np.sin(x / 16))

    def _nl(self, v: np.ndarray) -> np.ndarray:
        """the nonlinear term -u u_x = -(u^2)_x / 2, in spectral space"""
        return -0.5 * self._D * np.fft.rfft(np.fft.irfft(v, n=self.nx)**2)

    def step(self, x: np.ndarray) -> np.ndarray:
        v = np.fft.rfft(x)
        N1 = self._nl(v)
        v1 = self._E2 * v + self._Q * N1
        N2a = self._nl(v1)
        v2a = self._E2 * v + self._Q * N2a
        N2b = self._nl(v2a)
        v2b = self._E2 * v1 + self._Q * (2 * N2b - N1)
        N3 = self._nl(v2b)
        v = self._E * v + N1 * self._f1 + 2 * (N2a + N2b) * self._f2 + N3 * self._f3
        return np.fft.irfft(v, n=self.nx)
