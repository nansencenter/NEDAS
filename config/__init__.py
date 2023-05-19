##before starting python, make sure to
##   set -a; source config_file; set +a
##to export the env variables for a case

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
NX = int((XEND - XSTART) / DX)
NY = int((YEND - YSTART) / DX)
NZ_ICE = int(os.environ['NZ_ICE'])
NZ_ATM = int(os.environ['NZ_ATM'])
NZ_OCE = int(os.environ['NZ_OCE'])

##perturbation
PERTURB_PARAM_DIR = os.environ['PERTURB_PARAM_DIR']
PERTURB_NUM_SCALE = int(os.environ['PERTURB_NUM_SCALE'])
PERTURB_NUM_ENS = int(os.environ['PERTURB_NUM_ENS'])

##Observation

##DA parameters
NUM_ENS = int(os.environ['NUM_ENS'])


##multiscale
NUM_SCALE = int(os.environ['NUM_SCALE'])

