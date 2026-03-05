from NEDAS.config import Config
from NEDAS.schemes import get_scheme

config = Config(parse_args=True)

scheme = get_scheme(config)

scheme()
