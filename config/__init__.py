##before starting python, make sure to run bash command:
##  set -a; source <config_file>
##  to export the env variables, so that in python script
##  "import config as c" will get the env variables

###Parse system environment variables, most defined in config_file
###except for a few runtime variables defined in run scripts
import numpy as np
import os
import importlib
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
assim_mode = os.environ.get('assim_mode')
assert assim_mode in ('batch', 'serial'), 'unknown assimilation mode: '+assim_mode
filter_type = os.environ.get('filter_type')
###check type if valid
regress_type = os.environ.get('regress_type', 'linear')

run_assim = os.environ.get('run_assim').lower()=='true'
run_align_space = os.environ.get('run_align_space').lower()=='true'
run_align_time = os.environ.get('run_align_time').lower()=='true'

##multiscale
nscale = int(os.environ.get('nscale', '1'))
scale = int(os.environ.get('scale', '0'))  ##if not defined, use the first index
##for each scale component, analysis is stored in a separate s_dir
if nscale > 1:
    s_dir = f'scale_{scale+1}'
else:
    s_dir = ''

##localization
localize_type = os.environ.get('localize_type')
hroi_factor = np.array(os.environ.get('hroi_factor').split()).astype(np.float32)[scale]
vroi_factor = np.array(os.environ.get('vroi_factor').split()).astype(np.float32)[scale]
troi_factor = np.array(os.environ.get('troi_factor').split()).astype(np.float32)[scale]

##inflation
inflate_type = os.environ.get('inflate_type')
inflate_factor = np.array(os.environ.get('inflate_factor').split()).astype(np.float32)[scale]


##time control for experiment
time_start = os.environ.get('time_start')
time_end = os.environ.get('time_end')
time_assim_start = os.environ.get('time_assim_start')
time_assim_end = os.environ.get('time_assim_end')

cycle_period = np.float32(os.environ.get('cycle_period'))

time = os.environ.get('time', time_start)  ##time is the current analysis cycle
prev_time = os.environ.get('prev_time', time_start)
next_time = os.environ.get('next_time', time_start)

##time steps for the analysis window
obs_time_steps = np.array(os.environ.get('obs_time_steps', '0').split()).astype(np.float32)
state_time_steps = np.array(os.environ.get('state_time_steps', '0').split()).astype(np.float32)

##time multiscale smoothing
obs_time_scale = float(os.environ.get('obs_time_scale', '0'))
state_time_scale = float(os.environ.get('state_time_scale', '0'))


##define analysis grid
##TODO: not a good solution to load big file here in config, since it is imported by all other modules
grid_type = os.environ.get('grid_type')
if grid_type == 'new':
    proj = Proj(os.environ.get('proj'))
    dx = np.array(os.environ.get('dx').split()).astype(np.float32)[scale]
    xmin = np.float32(os.environ.get('xmin'))
    xmax = np.float32(os.environ.get('xmax'))
    ymin = np.float32(os.environ.get('ymin'))
    ymax = np.float32(os.environ.get('ymax'))
    grid = Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, centered=True)

    ##mask for nan area in domain, where no i/o or analysis tasks needed
    maskfile = os.environ.get('maskfile', '')
    if os.path.exists(maskfile):
        mask_dat = np.load(maskfile, allow_pickle=True)
        mask_grid = Grid(proj, mask_dat['x'], mask_dat['y'], dst_grid=grid)
        mask = (mask_grid.convert(mask_dat['mask'])==1)
        mask[-1, :] = True  ##TODO: something wrong along the border
        mask[:, -1] = True
        del mask_dat, mask_grid
    else:
        mask = np.full((grid.ny, grid.nx), False, dtype=bool)  ##no masked area by default

else:
    model = grid_type
    src = importlib.import_module('models.'+model)
    path = data_dir+'/'+model
    grid = src.read_grid(path)
    mask = src.read_mask(path, grid)

nx = grid.nx
ny = grid.ny


##parse state variables definition
state_def = []
lines = os.environ.get('state_def').split('\n')
for r in range(len(lines)):
    ss = lines[r].split()
    assert len(ss) == 4, 'state_def format error, should be "varname, source, var_type, err_type"'
    state_def.append({'name': ss[0], 'source': ss[1], 'state_type': ss[2], 'err_type': ss[3]})


##parse observation definition
obs_def = []
lines = os.environ.get('obs_def').split('\n')
for r in range(len(lines)):
    ss = lines[r].split()

    ##check line format is correct
    assert len(ss)==15, 'obs_def format error, should be "varname, source, model, obs_window_min, obs_window_max, err_type, err_std, err_hcorr, err_vcorr, err_tcorr, cross_corr[list of variable=corr_coef,...], hroi, vroi, troi, impact_on_state[list of state_variable=impact_factor,...]"'

    obs_def.append({'name': ss[0],
                    'source': ss[1],
                    'model': ss[2],
                    'obs_window_min': int(ss[3]),
                    'obs_window_max': int(ss[4]),
                    'err_type': ss[5],
                    'err_std': np.float32(ss[6]),
                    'err_hcorr': np.float32(ss[7]),
                    'err_vcorr': np.float32(ss[8]),
                    'err_tcorr': np.float32(ss[9]),
                    'hroi': np.float32(ss[11]),
                    'vroi': np.float32(ss[12]),
                    'troi': np.float32(ss[13]),
                    })

##update obs_impact_on_state
for r in range(len(lines)):
    ss = lines[r].split()

    ##default is 1 for all state variables
    obs_def[r]['impact_on_state'] = np.ones(len(state_def))

    for en in ss[14].split(','):
        if '=' in en:
            v, fac = en.split('=')
            ##check if v is one of the defined state variables
            p = [s for s,srec in enumerate(state_def) if srec['name']==v]
            assert len(p)>0, 'obs impact_on_state: '+v+' not defined in state_def'

            ##update the impact factor for this state variable
            obs_def[r]['impact_on_state'][p] = np.float32(fac)

##update obs_err_cross_corr
for r in range(len(lines)):
    ##default is uncorrelated (identity matrix)
    obs_def[r]['err_cross_corr'] = np.zeros(len(lines))
    obs_def[r]['err_cross_corr'][r] = 1

for r in range(len(lines)):
    ss = lines[r].split()
    name = ss[0]
    source = ss[1]

    for en in ss[10].split(','):
        if '=' in en:
            v, coef = en.split('=')
            ##check if v is one of the defined obs variables
            p = [s for s,srec in enumerate(obs_def) if srec['name']==v and srec['source']==source]
            assert len(p)==1, 'obs err_cross_corr: '+v+' either not defined in obs_def or there are duplicates with same name and source'
            p1 = p[0]
            coef1 = np.float32(coef)

            ##check if provided coef makes sense
            if p1 == r:
                assert coef1==1.0, 'obs err_cross_corr: corr btw '+v+' and itself should be 1, got '+coef
            if p1 < r:
                coef1old = obs_def[r]['err_cross_corr'][p1]
                assert coef1old==coef1, 'obs err_cross_corr: corr btw '+v+' and '+name+' is {}'.format(coef1old)+', got a new conflicting value '+coef
            obs_def[r]['err_cross_corr'][p1] = coef1
            obs_def[p1]['err_cross_corr'][r] = coef1


use_synthetic_obs = os.environ.get('use_synthetic_obs').lower()=='true'


##setup parallel scheme
from parallel import parallel_start
comm = parallel_start()

nproc = int(os.environ.get('nproc'))
nproc_mem = int(os.environ.get('nproc_mem'))
if comm.Get_size() == 1:
    ##likely run by jupyter, so we reset nproc_mem to avoid the check below
    nproc = 1
    nproc_mem = 1
else:
    assert nproc == comm.Get_size(), f"nproc {comm.Get_size()} is not the same as defined in config {nproc}"
    assert nproc % nproc_mem == 0, "nproc should be evenly divided by nproc_mem"
nproc_rec = int(nproc/nproc_mem)

pid = comm.Get_rank()  ##processor id

pid_mem = pid % nproc_mem
pid_rec = pid // nproc_mem
comm_mem = comm.Split(pid_rec, pid_mem)
comm_rec = comm.Split(pid_mem, pid_rec)

pid_show = 0  ##which pid is showing progress messages, default to root=0

debug = os.environ.get('debug').lower()=='true'

