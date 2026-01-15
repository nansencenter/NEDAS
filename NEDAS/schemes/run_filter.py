from NEDAS.config import Config
from NEDAS.schemes.get_analysis_scheme import get_analysis_scheme

c = Config(parse_args=True)
scheme = get_analysis_scheme(c)

scheme.filter(c)
