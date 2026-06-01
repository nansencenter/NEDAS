import numpy as np
from NEDAS.core import Context, Transform
from NEDAS.core.types import FieldRecord, ObsRecord

class Identity(Transform):
    """
    Subclass for the identity transform.
    """
    def forward_state(self, c: Context, rec: FieldRecord, field: np.ndarray) -> np.ndarray:
        return field

    def backward_state(self, c: Context, rec: FieldRecord, field: np.ndarray) -> np.ndarray:
        return field

    def forward_obs(self, c: Context, obs_rec: ObsRecord, obs_seq: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return obs_seq

    def backward_obs(self, c: Context, obs_rec: ObsRecord, obs_seq: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return obs_seq

