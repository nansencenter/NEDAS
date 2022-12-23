###parse config_file and get environment variables:
##in bash environment: "set -a; source config_file; set +a"
##this will export the settings into bash environment variables
##then, in python scripts do "import config.constants as cc"
##the variable "var" will be available as cc.var

import os

##experiment design
EXP_NAME=os.environ['EXP_NAME']
DATE_START=os.environ['DATE_START']
DATE_END=os.environ['DATE_END']
DATE_CYCLE_START=os.environ['DATE_CYCLE_START']
DATE_CYCLE_END=os.environ['DATE_CYCLE_END']
CYCLE_PERIOD=int(os.environ['CYCLE_PERIOD'])

##HPC settings, paths, env config
WORK_DIR=os.environ['WORK_DIR']
SCRIPT_DIR=os.environ['SCRIPT_DIR']
CODE_DIR=os.environ['CODE_DIR']
DATA_DIR=os.environ['DATA_DIR']

##reference grid definition
RE = float(os.environ['RE'])
ECC = float(os.environ['ECC'])
LON_0 = float(os.environ['LON_0'])
LAT_0 = float(os.environ['LAT_0'])
LAT_TS = float(os.environ['LAT_TS'])
DX = float(os.environ['DX'])
XSTART = float(os.environ['XSTART'])
XEND = float(os.environ['XEND'])
YSTART = float(os.environ['YSTART'])
YEND = float(os.environ['YEND'])
NX = int((XEND - XSTART) / DX)
NY = int((YEND - YSTART) / DX)
NZ_ICE = int(os.environ['NZ_ICE'])
NZ_ATM = int(os.environ['NZ_ATM'])
NZ_OCE = int(os.environ['NZ_OCE'])

##perturbation
PERTURB_PARAM_DIR = os.environ['PERTURB_PARAM_DIR']
PERTURB_NUM_SCALE = int(os.environ['PERTURB_NUM_SCALE'])

##Observation

##DA parameters
NUM_ENS = int(os.environ['NUM_ENS'])


##multiscale
NUM_SCALE = int(os.environ['NUM_SCALE'])

