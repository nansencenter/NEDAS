import os
import yaml
import importlib
import pyproj

from grid import Grid
from utils.parallel import parallel_start, bcast_by_root
from utils.log import message, timer

import config as c

def init_comm():
    ##initialize parallel communicator
    c.comm = parallel_start()

    ##if serial run, overwrite nproc
    # if c.comm.Get_size() == 1:
    #     c.nproc = 1
    #     c.nproc_mem = 1

    if hasattr(c, 'nproc'):
        assert c.nproc == c.comm.Get_size(), f"nproc {c.comm.Get_size()} is not the same as defined in config {c.nproc}"
    else:
        c.nproc = c.comm.Get_size()

    c.pid = c.comm.Get_rank() ##current processor id

    ##divide processors into mem/rec groups
    if not hasattr(c, 'nproc_mem'):
        c.nproc_mem = c.nproc
    assert c.nproc % c.nproc_mem == 0, "nproc {c.nproc} is not evenly divided by nproc_mem {c.nproc_mem}"
    c.nproc_rec = int(c.nproc/c.nproc_mem)

    c.pid_mem = c.pid % c.nproc_mem
    c.pid_rec = c.pid // c.nproc_mem
    c.comm_mem = c.comm.Split(c.pid_rec, c.pid_mem)
    c.comm_rec = c.comm.Split(c.pid_mem, c.pid_rec)

    c.pid_show = 0  ##which pid is showing progress messages, default to root=0


def init_grid():
    ##initialize analysis grid
    if c.grid_def['type'] == 'custom':
        proj = pyproj.Proj(c.grid_def['proj'])
        xmin, xmax = c.grid_def['xmin'], c.grid_def['xmax']
        ymin, ymax = c.grid_def['ymin'], c.grid_def['ymax']
        dx = c.grid_def['dx']
        c.grid = Grid.regular_grid(proj, xmin, xmax, ymin, ymax, dx, centered=True)

    else:
        ##get analysis grid from model module
        model_name = c.grid_def['type']
        model_src = importlib.import_module('models.'+model_name)
        model_dir = os.path.join(c.data_dir, model_name)
        c.grid = model_src.read_grid(model_dir)

    ##mask for invalid grid points
    # if c.mask


def init_config(config_file=None):

    ##parse the config_file to get a dict with variables
    ##define attributes c.variable=value
    config_dict = c.parse_config(config_file)
    for key, value in config_dict.items():
        setattr(c, key, value)

    ##host machine specific settings (expected in NEDAS/config/env/host)
    host_config_file = os.path.join(c.nedas_dir, 'config', 'env', c.host, 'base.yml')
    with open(host_config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            setattr(c, key, value)

    if not os.path.exists(c.work_dir):
        os.mkdir(c.work_dir)
    os.chdir(c.work_dir)

    init_comm()

    init_grid()

    message(c.comm, f'''Initializing config...
Working directory: {c.work_dir}
Parallel scheme: nproc = {c.nproc}, nproc_mem = {c.nproc_mem}
Analysis grid: type = {c.grid_def['type']}, shape = {c.grid.x.shape}
''', c.pid_show)

