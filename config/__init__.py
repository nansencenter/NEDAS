##before starting python, make sure to
##   set -a; source config_file; set +a
##to export the env variables

###Parse system environment variables for settings
import numpy as np
import os

##experiment design
EXP_NAME=os.environ['EXP_NAME']
DATE_START=os.environ['DATE_START']
DATE_END=os.environ['DATE_END']
DATE_CYCLE_START=os.environ['DATE_CYCLE_START']
DATE_CYCLE_END=os.environ['DATE_CYCLE_END']
CYCLE_PERIOD=np.float32(os.environ['CYCLE_PERIOD'])

##HPC settings, paths, env config
SCRATCH=os.environ['SCRATCH']
WORK_DIR=os.environ['WORK_DIR']
SCRIPT_DIR=os.environ['SCRIPT_DIR']
CODE_DIR=os.environ['CODE_DIR']
DATA_DIR=os.environ['DATA_DIR']

MASK_FROM = os.environ['MASK_FROM']
MESH_FILE = os.environ['MESH_FILE']
BATHY_FILE = os.environ['BATHY_FILE']

##reference grid definition
PROJ = os.environ['PROJ']
DX = np.float32(os.environ['DX'])
XSTART = np.float32(os.environ['XSTART'])
XEND = np.float32(os.environ['XEND'])
YSTART = np.float32(os.environ['YSTART'])
YEND = np.float32(os.environ['YEND'])

from grid import Grid
import pyproj
ref_grid = Grid.regular_grid(pyproj.Proj(PROJ), XSTART, XEND, YSTART, YEND, DX, centered=True)

##state variables definition
##  one state variable per line, each line contains:
##     variable name, string
##     source, string, which module takes care of preprocessing
##     dt, hours, how frequently state is available
##     zmin, zmax, int, vertical layer index start and end
STATE_DEF_FILE = os.environ['STATE_DEF_FILE']
ZI_MIN = int(os.environ['ZI_MIN'])
ZI_MAX = int(os.environ['ZI_MAX'])

##perturbation
PERTURB_PARAM_DIR = os.environ['PERTURB_PARAM_DIR']
PERTURB_NUM_SCALE = int(os.environ['PERTURB_NUM_SCALE'])
PERTURB_NUM_ENS = int(os.environ['PERTURB_NUM_ENS'])

##Observation
OBS_DEF_FILE = os.environ['OBS_DEF_FILE']
##obs variables definition
##file structure:
##  one obs variable per line, each line contains:
##     variable name,  string, see obs_def.variables for available names
##     source, string, which module takes care of preprocessing
##     error model, string, to specify obs error
##     horizontal ROI,   real, localization radius of influence, in meters
##     vertical ROI,     real, vertical .. .. .., in meters
##     impact factor on a list of state_def.variables:
##        state_var_name=factor, factor is real number between 0 and 1.
obs_def = {}
with open(OBS_DEF_FILE, 'r') as f:
    for lin in f.readlines():
        entry = lin.split()
        vname = entry[0]
        source = entry[1]
        error = np.float32(entry[2])
        hroi = np.float32(entry[3])
        vroi = np.float32(entry[4])
        impact = {}
        for ilin in entry[5:]:
            impact_vname, impact_factor = ilin.split('=')
            impact.update({impact_vname:np.float32(impact_factor)})
        obs_def.update({vname:{'source':source,
                            'error':error,
                            'hroi':hroi,
                            'vroi':vroi,
                            'impact':impact}})

OBS_WINDOW_MIN=np.float32(os.environ['OBS_WINDOW_MIN'])
OBS_WINDOW_MAX=np.float32(os.environ['OBS_WINDOW_MAX'])

##DA parameters
NUM_ENS = int(os.environ['NUM_ENS'])

##multiscale
NUM_SCALE = int(os.environ['NUM_SCALE'])


