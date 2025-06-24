import numpy as np
from NEDAS.config import Config
from .base import Transform

class Identity(Transform):
    """
    Subclass for the identity transform.
    """
    def forward_state(self, c: Config, rec: dict, field: np.ndarray) -> np.ndarray:
        return field

    def backward_state(self, c: Config, rec: dict, field: np.ndarray) -> np.ndarray:
        return field

    def forward_obs(self, c: Config, obs_rec: dict, obs_seq: dict) -> dict:
        return obs_seq

    def backward_obs(self, c: Config, obs_rec: dict, obs_seq: dict) -> dict:
        return obs_seq

