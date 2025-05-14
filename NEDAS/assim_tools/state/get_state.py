from NEDAS.config import Config
from .base import State

registry = {
}

def get_state(c: Config) -> State:
    #if state_type not in registry.keys():
    #    return State
    #module = importlib.import_module('NEDAS.assim_tools.state.'+state_type)
    #StateClass = getattr(module, registry[state_type])
    #return StateClass
    return State(c)
