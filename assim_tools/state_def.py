##state variable definition
##available state variable names, and their properties
variables = {'atmos_surf_wind': {'is_vector':True, 'units':'m/s'},
             'atmos_surf_temp': {'is_vector':False, 'units':'K'},
             'atmos_surf_dew_temp': {'is_vector':False, 'units':'K'},
             'atmos_surf_press': {'is_vector':False, 'units':'Pa'},
             'atmos_precip': {'is_vector':False, 'units':'kg/m2/s'},
             'atmos_snowfall': {'is_vector':False, 'units':'kg/m2/s'},
             'atmos_down_shortwave': {'is_vector':False, 'units':'W/m2'},
             'atmos_down_longwave': {'is_vector':False, 'units':'W/m2'},
             'seaice_conc': {'is_vector':False, 'units':'%'},
             'seaice_thick': {'is_vector':False, 'units':'m'},
             'snow_thick': {'is_vector':False, 'units':'m'},
             'seaice_drift': {'is_vector':True, 'units':'m'},
             'seaice_velocity': {'is_vector':True, 'units':'m/s'},
             'seaice_damage': {'is_vector':False, 'units':'%'},
             'ocean_surf_temp': {'is_vector':False, 'units':'K'},
             'ocean_surf_velocity': {'is_vector':True, 'units':'m/s'},
             'ocean_surf_height': {'is_vector':False, 'units':'m'},
             'ocean_temp': {'is_vector':False, 'units':'K'},
             'ocean_salinity': {'is_vector':False, 'units':'%'},
             'ocean_velocity': {'is_vector':True, 'units':'m/s'},
             }

###units converter
##units: target units used in state variables (defined in variables)
##s_units: source units used in models (defined in var_dict in each module)
def units_convert(units, s_units, var, inverse=False):
    ##  list of available units and methods to convert from/to s_units
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


