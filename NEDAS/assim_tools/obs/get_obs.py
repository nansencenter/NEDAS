from NEDAS.config import Config
from NEDAS.assim_tools.state import State
from .base import Obs

def get_obs(c: Config, state: State) -> Obs:
    # obs_type = [r['type'] for r in c.obs_def]
    # module = importlib.import_module('NEDAS.assim_tools.obs.'+obs_type)
    # ObsClass = getattr(module, registry[obs_type])
    # return ObsClass(c, state)
    return Obs(c, state)
