##before starting python, make sure to run bash command:
##  set -a; source <config_file>
##  to export the env variables, so that in python script
##  "import config as c" will get the env variables

###Parse system environment variables, most defined in config_file
###except for a few runtime variables defined in run scripts
import numpy as np
import os
from grid import Grid
from pyproj import Proj

exp_name = os.environ.get('exp_name')

##directories
script_dir = os.environ.get('script_dir')
code_dir = os.environ.get('code_dir')
data_dir = os.environ.get('data_dir')
work_dir = os.environ.get('work_dir')

##ensemble size
nens = int(os.environ.get('nens'))

##DA scheme
run_filter = os.environ.get('run_filter').lower()=='true'
run_align_space = os.environ.get('run_align_space').lower()=='true'
run_align_time = os.environ.get('run_align_time').lower()=='true'

filter_type = os.environ.get('filter_type')

##multiscale
nscale = int(os.environ.get('nscale'))
scale = int(os.environ.get('scale', '0'))  ##if not defined, use the first index

##localization
localize_type = os.environ.get('localize_type')
hroi_factor = np.array(os.environ.get('hroi_factor').split()).astype(np.float32)[scale]
vroi_factor = np.array(os.environ.get('vroi_factor').split()).astype(np.float32)[scale]
troi_factor = np.array(os.environ.get('troi_factor').split()).astype(np.float32)[scale]

##inflation
inflate_type = os.environ.get('inflate_type')
inflate_factor = np.array(os.environ.get('inflate_factor').split()).astype(np.float32)[scale]


##time control for experiment
time_start=os.environ.get('time_start')
time_end=os.environ.get('time_end')
time_assim_start=os.environ.get('time_assim_start')
time_assim_end=os.environ.get('time_assim_end')

cycle_period=np.float32(os.environ.get('cycle_period'))

time=os.environ.get('time', time_start)  ##time is the current analysis cycle
prev_time=os.environ.get('prev_time', time_start)
next_time=os.environ.get('next_time', time_start)

##time scheme for analysis window
obs_window_min = np.float32(os.environ.get('obs_window_min'))
obs_window_max = np.float32(os.environ.get('obs_window_max'))
obs_nt = int(os.environ.get('obs_nt'))
obs_dt = (obs_window_max - obs_window_min) / obs_nt
obs_ts = np.arange(obs_window_min + obs_dt/2, obs_window_max, obs_dt)

##time scale for averaging
t_scale = np.array(os.environ.get('t_scale').split()).astype(np.float32)[scale]

##analysis time slots, typically only [0] at model restart time
if run_align_time:
    state_ts = obs_ts
else:
    state_ts = np.array([0])


##define analysis grid
proj = Proj(os.environ.get('proj'))
dx = np.array(os.environ.get('dx').split()).astype(np.float32)[scale]
xmin = np.float32(os.environ.get('xmin'))
xmax = np.float32(os.environ.get('xmax'))
ymin = np.float32(os.environ.get('ymin'))
ymax = np.float32(os.environ.get('ymax'))
nx = int((xmax - xmin) / dx)
ny = int((ymax - ymin) / dx)

grid = Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, centered=True)

##mask for nan area in domain, where no i/o or analysis tasks needed
maskfile = os.environ.get('maskfile')
if os.path.exists(maskfile):
    mask_dat = np.load(maskfile, allow_pickle=True)
    mask_grid = Grid(proj, mask_dat['x'], mask_dat['y'], dst_grid=grid)
    mask = (mask_grid.convert(mask_dat['mask'])==1)
    del mask_dat, mask_grid
else:
    mask = np.full((ny, nx), False, dtype=bool)  ##no masked area by default


##parse state variables definition
state_def = {}
for line in os.environ.get('state_def').split('\n'):
    ss = line.split()
    assert len(ss) == 3, 'state_def format error, should be "varname, source, err_type"'
    vname = ss[0]
    state_def[vname] = {'source': ss[1],
                        'err_type': ss[2],}


##parse observation definition
obs_def = {}
for line in os.environ.get('obs_def').split('\n'):
    ss = line.split()
    assert len(ss)>=8, 'obs_def format error, should be "varname, source, model, err_type, err, hroi, vroi, troi, impact_factor[list of state=factor, optional]"'
    vname = ss[0]
    obs_def[vname] = {'source': ss[1],
                      'model': ss[2],
                      'err_type': ss[3],
                      'err': np.float32(ss[4]),
                      'hroi': np.float32(ss[5]),
                      'vroi': np.float32(ss[6]),
                      'troi': np.float32(ss[7]),
                      'impact': {}, }
    ##observation impact factor dict[obs_variable][state_variable]
    ##first make a list of state variable and default impact as 1.
    impact = {}
    for v in state_def:
        impact[v] = 1.
    for en in ss[8:]:
        v, fac = en.split('=')
        if v in state_def:
            impact[v] = np.float32(fac)
    obs_def[vname]['impact'] = impact

use_synthetic_obs = os.environ.get('use_synthetic_obs').lower()=='true'


##clean up
del np,os,Proj,Grid,vname,v,fac,ss,en,

