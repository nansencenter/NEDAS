import numbers
import numpy as np

def units_convert(units_from, units_to, var):
    """
    units converter function

    Inputs:
    - units_from: str
      Source units for the input variable var

    - units_to: str
      Target units to convert var to

    - var: np.array
      The input variable

    Return:
    - var: np.array
      Variable with converted units
    """
    # if input units are numerics just apply the scaling factor
    if isinstance(units_to, numbers.Number) and isinstance(units_from, numbers.Number):
        return var * units_to / units_from

    # Define unit groups with a common base unit
    unit_groups = {
        "speed": {
            "base": "m/s",
            "to_base": {
                "km/h": lambda x: x / 3.6,
                "km/day": lambda x: x / 86.4,
                "cm/s": lambda x: x / 100.,
            },
            "from_base": {
                "km/h": lambda x: x * 3.6,
                "km/day": lambda x: x * 86.4,
                "cm/s": lambda x: x * 100.,
            },
        },
        "length": {
            "base": "m",
            "to_base": {
                "mm": lambda x: x / 1000.,
                "cm": lambda x: x / 100.,
                "dm": lambda x: x / 10.,
                "km": lambda x: x * 1000.,
            },
            "from_base": {
                "mm": lambda x: x * 1000.,
                "cm": lambda x: x * 100.,
                "dm": lambda x: x * 10.,
                "km": lambda x: x / 1000.,
            },
        },
        "time": {
            "base": "s",
            "to_base": {
                "min": lambda x: x * 60.,
                "h": lambda x: x * 3600.,
                "day": lambda x: x * 86400.,
            },
            "from_base": {
                "min": lambda x: x / 60.,
                "h": lambda x: x / 3600.,
                "day": lambda x: x / 86400.,
            },
        },
        "weight": {
            "base": "g",
            "to_base": {
                "kg": lambda x: x * 1000.,
            },
            "from_base": {
                "kg": lambda x: x / 1000.,
            },
        },
        "temperature": {
            "base": "K",
            "to_base": {
                "C": lambda x: x + 273.15,
                "F": lambda x: (x - 32) * 5 / 9 + 273.15,
            },
            "from_base": {
                "C": lambda x: x - 273.15,
                "F": lambda x: (x - 273.15) * 9 / 5 + 32,
            },
        },
        "pressure": {
            "base": "Pa",
            "to_base": {
                "hPa": lambda x: x * 100,
                "bar": lambda x: x * 100000.,
                "mbar": lambda x: x * 100.,
            },
            "from_base": {
                "hPa": lambda x: x / 100.,
                "bar": lambda x: x / 100000., 
                "mbar": lambda x: x / 100.,
            },
        },
        "energy_flux": {
            "base": "W/m2",
            "to_base": {
                "J/m2/d": lambda x: x / 86400.,
            },
            "from_base": {
                "J/m2/d": lambda x: x * 86400.,
            },
        },
        # Add other groups here...
    }

    if units_to == units_from:
        return var
    
    # Find the group containing the units
    for group, definitions in unit_groups.items():
        base = definitions["base"]
        to_base = definitions["to_base"]
        from_base = definitions["from_base"]

        # Check if both units exist in this group
        if units_from == base:
            if units_to in from_base:
                return from_base[units_to](var)
        elif units_to == base:
            if units_from in to_base:
                return to_base[units_from](var)
        elif units_from in to_base and units_to in from_base:
            # Convert to base, then from base to target
            var_base = to_base[units_from](var)
            return from_base[units_to](var_base)    

    raise ValueError(f"Conversion of unit from '{units_from}' to '{units_to}' not supported.")


##binary file io type conversion
type_convert = {'double':np.float64, 'float':np.float32, 'int':np.int32}
type_dic = {'double':'d', '8':'d', 'single':'f', 'float':'f', '4':'f', 'int':'i'}
type_size = {'double':8, 'float':4, 'int':4}

##map projection name in pyproj
from pyproj import Proj
def proj2dict(proj:Proj) -> dict:
    proj_names = {
        "stere": "stereographic",
        "merc": "mercator",
        "lcc": "lambert_conformal_conic",
        "utm": "transverse_mercator",
        "aea": "albers_conical_equal_area",
        "eqc": "equirectangular",
        }
    params = {
        "proj":  {'name':"projection", 'type':str},
        "datum": {'name':"datum", 'type':str},
        "R":     {'name':"earth_radius", 'type':float},
        "lat_0": {'name':"latitude_of_projection_origin", 'type':float},
        "lon_0": {'name':"longitude_of_projection_origin", 'type':float},
        "lat_ts": {'name':"standard_parallel", 'type':float},
        "lat_1": {'name':"first_standard_parallel", 'type':float},
        "lat_2": {'name':"second_standard_parallel", 'type':float},
        "x_0": {'name':"false_easting", 'type':float},
        "y_0": {'name':"false_northing", 'type':float},
        "a": {'name':"semi_major_axis", 'type':float},
        "b": {'name':"semi_minor_axis", 'type':float},
        "k": {'name':"scale_factor_at_projection_origin", 'type':float},
    }
    proj_params = {}
    for entry in proj.definition.split():
        if '=' in entry:
            key, value = entry.split('=', 1)
            if key in params:
                value = params[key]['type'](value)
        else:
            key, value = entry, True
        if key in params:
            if key == 'proj' and value in proj_names:
                value = proj_names[value]
            proj_params[params[key]['name']] = value
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

