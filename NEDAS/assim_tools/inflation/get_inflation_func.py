from NEDAS.config import Config
from .base import Inflation

registry = {
    'multiplicative': 'MultiplicativeInflation',
    'RTPP': 'RTPPInflation',
}

def get_inflation_func(c: Config) -> Inflation:
    if c.inflation:
        inflation_type = c.inflation.get('type', '').split(',')
        if 'multiplicative' in inflation_type:
            from .multiplicative import MultiplicativeInflation as Inflation
        elif 'RTPP' in inflation_type:
            from .RTPP import RTPPInflation as Inflation
        else:
            from .base import Inflation
    else:
        from .base import Inflation
    return Inflation(c)