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
                             'mbar':lambda x: x*100.,
                            },
                 'kg/m2/s': {'Mg/m2/3h':lambda x: x/3/3.6,
                            },
                 'W/m2':    {'J/m2/6h':lambda x:x,
                            },
                 'precip m/s': {'precip m/6h':lambda x:x,
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
                             'mbar':lambda x: x/100.,
                            },
                 'kg/m2/s': {'Mg/m2/3h':lambda x: x*3*3.6,
                            },
                 'W/m2':    {'J/m2/6h':lambda x:x,
                            },
                 'precip m/s': {'precip m/6h':lambda x:x,
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

##map projection name in pyproj
from pyproj import Proj
def proj2dict(proj:Proj) -> dict:
    proj_names = {
        "stere": "polar_stereographic",
        "merc": "mercator",
        "lcc": "lambert_conformal_conic",
        "utm": "transverse_mercator",
        "aea": "albers_conical_equal_area",
        "eqc": "equirectangular",
        }
    param_names = {
        "proj": "projection",
        "datum": "datum",
        "lat_0": "central_latitude",
        "lon_0": "central_longitude",
        "lat_ts": "standard_parallel",
        "x_0": "false_easting",
        "y_0": "false_northing",
        "a": "semi_major_axis",
        "b": "semi_minor_axis",
        "k": "scale_factor_at_origin",
    }
    proj_params = {}
    for entry in proj.definition.split():
        if '=' in entry:
            key, value = entry.split('=', 1)
            if value.replace('.','',1).isdigit():
                value = float(value)
        else:
            key, value = entry, True
        if key in param_names:
            proj_params[param_names[key]] = value
    return proj_params

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

def seconds_to_timestr(seconds):
    """convert from seconds to time duration string"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def ensure_list(v):
    if isinstance(v, list):
        return v
    return [v]

