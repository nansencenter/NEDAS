import numpy as np
from NEDAS.models.ode_model import OdeModel


class Lorenz96TwoScaleModel(OdeModel):
    """
    The two-scale Lorenz (1996) model: K slow variables X, each coupled to J fast variables Y.

    As in DART models/lorenz_96_2scale (Lorenz's version, the Y form one periodic chain), and
    DAPPER mods/LorenzUV. The state vector is [X, Y] as in DART. NEDAS carries both on the grid of
    the J*K fast variables, as the variables 'slow' (X) and 'fast' (Y): X_k is over the J points
    of its Y_{j,k}. Reading 'slow' repeats X_k over them, writing it adds the mean change over them
    to X_k (so an unchanged field is written back exactly).

    Args:
        K, J (int): number of slow variables, and of fast variables per slow one
        F (float): forcing
        coupling_b, coupling_c, coupling_h (float): the spatial and time scale ratios of the fast
            variables, and the coupling strength (b, c, h of Lorenz 1996)
    """
    cyclic = True
    K: int
    J: int
    F: float
    coupling_b: float
    coupling_c: float
    coupling_h: float

    @property
    def state_size(self) -> int:
        return self.K * (self.J + 1)

    def grid_size(self) -> int:
        return self.K * self.J

    def field_names(self) -> list[str]:
        return ['slow', 'fast']

    def x0(self) -> np.ndarray:
        x = np.zeros(self.state_size)
        x[:self.K] = self.F
        x[0] += 1.  # off the fixed point
        return x

    def dxdt(self, x: np.ndarray) -> np.ndarray:
        K, J, b, c, h = self.K, self.J, self.coupling_b, self.coupling_c, self.coupling_h
        X, Y = x[:K], x[K:]
        dX = (np.roll(X, -1) - np.roll(X, 2)) * np.roll(X, 1) - X + self.F - h * c / b * Y.reshape(K, J).sum(axis=1)
        dY = c * b * np.roll(Y, -1) * (np.roll(Y, 1) - np.roll(Y, -2)) - c * Y + h * c / b * np.repeat(X, J)
        return np.concatenate([dX, dY])

    def get_field(self, x: np.ndarray, name: str) -> np.ndarray:
        if name == 'slow':
            return np.repeat(x[:self.K], self.J)
        if name == 'fast':
            return x[self.K:].copy()
        raise ValueError(f"unknown variable '{name}'")

    def set_field(self, x: np.ndarray, name: str, fld: np.ndarray) -> None:
        if name == 'slow':
            x[:self.K] += (fld - np.repeat(x[:self.K], self.J)).reshape(self.K, self.J).mean(axis=1)
        elif name == 'fast':
            x[self.K:] = fld
        else:
            raise ValueError(f"unknown variable '{name}'")
