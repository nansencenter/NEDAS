import numpy as np

def units_convert(units, s_units, var, inverse=False):
    """
    units converter function

    Inputs:
    - units: str
      Target units used in analysis/observation

    - s_units: str
      Source units from native model variables

    - var: np.array
      The input variable defined with s_units

    - inverse: bool, optional
      Whether the conversion is inverse, from units to s_units. Default is False.

    Return:
    - var: np.array
      Variable with converted units
    """
    ## dict[units][s_units] for methods to convert units from/to s_units
    unit_from = {'m/s':     {'km/h':lambda x: x/3.6,
                             'km/day':lambda x: x/86.4,
                            },
                 'm':       {'cm':lambda x: x/100.,
                            },
                 'K':       {'C':lambda x: x+273.15,
                             'F':lambda x: (x-32)*5./9.+273.15,
                            },
                 'Pa':      {'hPa':lambda x: x*100.,
                            },
                 'kg/m2/s': {'Mg/m2/3h':lambda x: x/3/3.6,
                            },
                }
    unit_to   = {'m/s':     {'km/h':lambda x: x*3.6,
                             'km/day':lambda x: x*86.4,
                            },
                 'm':       {'cm':lambda x: x*100.,
                            },
                 'K':       {'C':lambda x: x-273.15,
                             'F':lambda x: (x-273.15)*9./5.+32.,
                            },
                 'Pa':      {'hPa':lambda x: x/100.,
                            },
                 'kg/m2/s': {'Mg/m2/3h':lambda x: x*3*3.6,
                            },
                }
    if units != s_units:
        if s_units not in unit_from[units]:
            raise ValueError("cannot find convert method from "+s_units+" to "+units)
        else:
            if not inverse:
                var = unit_from[units][s_units](var)
            else:
                var = unit_to[units][s_units](var)
    return var

##binary file io type conversion
type_convert = {'double':np.float64, 'float':np.float32, 'int':np.int32}
type_dic = {'double':'d', '8':'d', 'single':'f', 'float':'f', '4':'f', 'int':'i'}
type_size = {'double':8, 'float':4, 'int':4}

from datetime import datetime, timedelta

dt1h = timedelta(hours=1)

def t2h(t):
    """convert datetime obj to hours since 1900-1-1 00:00"""
    return (t - datetime(1900,1,1))/timedelta(hours=1)

def h2t(h):
    """convert hours since 1900-1-1 00:00 to datetime obj"""
    return datetime(1900,1,1) + timedelta(hours=1) * h

def t2s(t):
    """convert datetime obj to a time string 'ccyymmddHHMM'"""
    return t.strftime('%Y%m%d%H%M')

def s2t(s):
    """convert a time string 'ccyymmddHHMM' to a datetime obj"""
    return datetime.strptime(s, '%Y%m%d%H%M')


def ensure_list(v):
    if isinstance(v, list):
        return v
    return [v]

