import numpy as np
import glob
from datetime import datetime, timedelta

variables = {'': {'name':'', 'dtype':'float', 'is_vector':False, 'restart_dt':1, 'levels':levels, 'units':''},
            }


def filename(path, **kwargs):

    return path


def read_grid(path, **kwargs):

    return grid


def write_grid(path, grid, **kwargs):
    pass


def read_var(path, grid, **kwargs):
    return var


def write_var(path, grid, var, **kwargs):
    pass


uniq_z_key = ()
def z_coords(path, grid, **kwargs):
    return z


