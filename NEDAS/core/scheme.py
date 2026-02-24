
from NEDAS.config import Config
from .coordinator import Coordinator

class Scheme:

    c: Coordinator

    def __init__(self, cf: Config):

        self.c = Coordinator(cf)

    def __call__(self):
        pass
