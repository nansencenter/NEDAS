import importlib
from typing import Type
from .base import State

registry = {
}

def get_state_class(state_type: str) -> Type['State']:
    if state_type not in registry.keys():
        return State
    module = importlib.import_module('NEDAS.assim_tools.state.'+state_type)
    StateClass = getattr(module, registry[state_type])
    return StateClass

