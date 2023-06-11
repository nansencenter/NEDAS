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
CYCLE_PERIOD=int(os.environ['CYCLE_PERIOD'])

##HPC settings, paths, env config
SCRATCH=os.environ['SCRATCH']
WORK_DIR=os.environ['WORK_DIR']
SCRIPT_DIR=os.environ['SCRIPT_DIR']
CODE_DIR=os.environ['CODE_DIR']
DATA_DIR=os.environ['DATA_DIR']

##reference grid definition
PROJ = os.environ['PROJ']
DX = float(os.environ['DX'])
XSTART = float(os.environ['XSTART'])
XEND = float(os.environ['XEND'])
YSTART = float(os.environ['YSTART'])
YEND = float(os.environ['YEND'])

from grid import Grid, regular_grid
import pyproj
ref_proj = pyproj.Proj(PROJ)
ref_x, ref_y = regular_grid(XSTART, XEND, YSTART, YEND, DX, centered=True)
ref_grid = Grid(ref_proj, ref_x, ref_y)

##state variables
STATE_DEF_FILE = os.environ['STATE_DEF_FILE']
##state variables definition
##  one state variable per line, each line contains:
##     variable name, string
##     source, string, which module takes care of preprocessing
##     nz, int, number of vertical layers
##     nt, int, number of time slices
state_def = {}
with open(STATE_DEF_FILE, 'r') as f:
    for lin in f.readlines():
        vname, source, nz, nt = lin.split()
        state_def.update({vname:{'source':source,
                                'nz':int(nz),
                                'nt':int(nt)}})

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


##DA parameters
NUM_ENS = int(os.environ['NUM_ENS'])

##multiscale
NUM_SCALE = int(os.environ['NUM_SCALE'])


