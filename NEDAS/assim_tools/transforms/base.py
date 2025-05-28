from abc import ABC, abstractmethod
import numpy as np
from NEDAS.config import Config

class Transform(ABC):
    """
    Base class for miscellaneous transform functions
    """
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def forward_state(self, c: Config, rec: dict, field: np.ndarray) -> np.ndarray:
        """
        Forward transform for the model state variables.

        Args:
            c (Config): Configuration object.
            rec (dict): State information record for the variable.
            field (np.ndarray): The state variable.

        Returns:
            np.ndarray: The transformed state variable.
        """
        return field

    @abstractmethod
    def backward_state(self, c: Config, rec: dict, field: np.ndarray) -> np.ndarray:
        """
        Backward (inverse) transform for the model state variables.

        Args:
            c (Config): Configuration object.
            rec (dict): State information record for the variable.
            field (np.ndarray): The transformed state variable.

        Returns:
            np.ndarray: The state variable transformed back to the original space.
        """
        return field

    @abstractmethod
    def forward_obs(self, c: Config, obs_rec: dict, obs_seq: dict) -> dict:
        """
        Forward transform for the observation sequence.

        Args:
            c (Config): Configuration object.
            obs_rec (dict): Observation information record.
            obs_seq (dict): The observation sequence.
                With keys ``'obs'`` the observation;
                ``'x', 'y', 'z', 't'`` the space/time coordinates;
                and ``'err_std'`` the observation errors.

        Returns:
            dict: The transformed observation sequence.
        """
        return obs_seq

    @abstractmethod
    def backward_obs(self, c: Config, obs_rec: dict, obs_seq: dict) -> dict:
        """
        Backward (inverse) transform for the observation sequence.

        Args:
            c (Config): Configuration object.
            obs_rec (dict): Observation information record.
            obs_seq (dict): The transformed observation sequence.
                With keys ``'obs'`` the observation;
                ``'x', 'y', 'z', 't'`` the space/time coordinates;
                and ``'err_std'`` the observation errors.

        Returns:
            dict: The observation sequence transformed back to the original space.
        """
        return obs_seq

