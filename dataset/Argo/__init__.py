import numpy as np
import importlib
import glob
from datetime import datetime, timedelta
import pyproj

###
###load bathymetry data for profile depth limits and mask land area.

variables = {'ocean_temp': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
             'ocean_saln': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'psu'},
             }

def filename(path, **kwargs):
    pass


def read_obs(path, **kwargs):

    return obs


##note: grid is the reference c.grid
def random_network(grid, mask):

    nprofile = 1000
    dz = 50 ##m

    obs_seq = {}

    obs_id = 0
    for i in range(nprofile):
        ##randomly find a point in unmasked (ocean) area:
        tmp = 1.
        while tmp > 0:
            x = np.random.uniform(grid.xmin, grid.xmax)
            y = np.random.uniform(grid.ymin, grid.ymax)
            tmp = grid.interp(mask.astype(int), x, y)

        nlevel = 1 #np.random.randint(10,50) ##nlevel differs for each profile

        z = 3 ##m
        for k in range(nlevel):
            rec = {'z': z, 'y': y, 'x': x}
            obs_seq[obs_id] = rec
            obs_id += 1
            z += dz

    return obs_seq


##convert state variable to observation variable (part of the obs-operator)
##modelpath: where the state variable can be read from using models.model.read_var
##grid: the target output grid, on which the obs_seq coords are defined
##kwargs: name, time, member, k
def state_to_obs(modelpath, grid, **kwargs):
    modelsrc = importlib.import_module('models.'+kwargs['model'])
    modelgrid = modelsrc.read_grid(modelpath, **kwargs)
    modelgrid.dst_grid = grid
    modelvar = modelsrc.read_var(modelpath, modelgrid, **kwargs)
    var = modelgrid.convert(modelvar, is_vector=kwargs['is_vector'])

    return var


