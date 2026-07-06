from abc import ABC, abstractmethod
import numpy as np
from .types import FieldRecord, ObsRecord
from .context import Context

class Transform(ABC):
    """
    Base class for miscellaneous transform functions
    """
    #: Whether forward_state/forward_obs are true no-ops given the current config (e.g. a
    #: ScaleBandpass transform with only 1 scale). When True for every active transform_func,
    #: code that needs H(X) (e.g. computing obs_prior/obs_post) may safely shortcut by reading
    #: cached state fields instead of re-reading the raw, untransformed model state -- otherwise
    #: it must always re-read the raw state, since the cached fields no longer represent it.
    #: Subclasses must set this correctly (statically or in __init__ based on config), not the
    #: caller -- only the transform itself knows whether its own forward_state is a no-op.
    is_identity: bool = False

    def __init__(self, c: Context, **kwargs) -> None:
        ...

    @abstractmethod
    def forward_state(self, c: Context, rec: FieldRecord, field: np.ndarray) -> np.ndarray:
        """
        Forward transform for the model state variables.

        Args:
            c (Context): the runtime context.
            rec (FieldRecord): State information record for the variable.
            field (np.ndarray): The state variable.

        Returns:
            np.ndarray: The transformed state variable.
        """
        ...

    @abstractmethod
    def backward_state(self, c: Context, rec: FieldRecord, field: np.ndarray) -> np.ndarray:
        """
        Backward (inverse) transform for the model state variables.

        Args:
            c (Context): the runtime context.
            rec (FieldRecord): State information record for the variable.
            field (np.ndarray): The transformed state variable.

        Returns:
            np.ndarray: The state variable transformed back to the original space.
        """
        ...

    @abstractmethod
    def forward_obs(self, c: Context, obs_rec: ObsRecord, obs_seq: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Forward transform for the observation sequence.

        Args:
            c (Context): the runtime context.
            obs_rec (ObsRecord): Observation information record.
            obs_seq (dict[str, np.ndarray]): The observation sequence.
                With keys ``'obs'`` the observation;
                ``'x', 'y', 'z', 't'`` the space/time coordinates;
                and ``'err_std'`` the observation errors.

        Returns:
            dict[str, np.ndarray]: The transformed observation sequence.
        """
        ...

    @abstractmethod
    def backward_obs(self, c: Context, obs_rec: ObsRecord, obs_seq: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Backward (inverse) transform for the observation sequence.

        Args:
            c (Context): the runtime context.
            obs_rFieldRecord)ict): Observation information record.
            obs_seq (dict): The transformed observation sequence.
                With keys ``'obs'`` the observation;
                ``'x', 'y', 'z', 't'`` the space/time coordinates;
                and ``'err_std'`` the observation errors.

        Returns:
            dict: The observation sequence transformed back to the original space.
        """
        ...

