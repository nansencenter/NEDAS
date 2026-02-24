from abc import ABC, abstractmethod

from NEDAS.config import Config
from .coordinator import Coordinator

class Scheme(ABC):

    c: Coordinator

    def __init__(self, cf: Config):

        self.c = Coordinator(cf)

    @abstractmethod
    def __call__(self) -> None:
        pass
