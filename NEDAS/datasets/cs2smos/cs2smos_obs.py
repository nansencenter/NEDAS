import os
import glob
import numpy as np
import pyproj
from datetime import datetime, timedelta, timezone
import netCDF4
from NEDAS.grid import Grid
from NEDAS.core import Dataset
from NEDAS.core.types import VarDesc

class Cs2SmosObs(Dataset):
    proj: str
    xstart: float
    xend: float
    ystart: float
    yend: float
    dx: float
    dy: float
    obs_file_dt: int
    obs_days: int
    use_dataset_uncertainty: bool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.variables = {
            'seaice_thick': VarDesc(name='analysis_sea_ice_thickness', dtype='float', is_vector=False, dt=24, levels=np.array([0]), z_units='m', units='m'),
            'seaice_conc': VarDesc(name='sea_ice_concentration', dtype='float', is_vector=False, dt=24, levels=np.array([0]), z_units='m', units=100),
            'seaice_type': VarDesc(name='sea_ice_type', dtype='int', is_vector=False, dt=24, levels=np.array([0]), z_units='m', units=1),
        }

        proj = pyproj.Proj(self.proj)
        x, y = np.meshgrid(np.arange(self.xstart, self.xend, self.dx), np.arange(self.ystart, self.yend, self.dy))
        self.grid = Grid(proj, x, y)

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        path = kwargs['path']
        time = kwargs['time']
        name = kwargs['name']
        obs_window_min = kwargs['obs_window_min']
        obs_window_max = kwargs['obs_window_max']

        search = ''
        if time is None:
            search = os.path.join(path, "W_XX-ESA,SMOS_CS2,NH_25KM_EASE2_????????_????????_r_v206_01_l4sit.nc")
            file_list = glob.glob(search)
        else:
            if obs_window_min is not None and obs_window_max is not None:
                d_range = np.arange(obs_window_min, obs_window_max, self.obs_file_dt)
            else:
                d_range = [0]

            file_list = []
            for d in d_range:
                t = time + d * timedelta(hours=1)
                obs_d = (self.obs_days - 1 ) // 2
                t1 = t - obs_d * timedelta(days=1)
                t2 = t + obs_d * timedelta(days=1)
                search = os.path.join(path, f"W_XX-ESA,SMOS_CS2,NH_25KM_EASE2_{t1:%Y%m%d}_{t2:%Y%m%d}_r_v206_01_l4sit.nc")
                for result in glob.glob(search):
                    if result not in file_list:
                        file_list.append(result)
        return file_list

    def generate_obs_network(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)

        if kwargs['nobs'] is None:
            nobs = 1000
        else:
            nobs = kwargs['nobs']

        grid = kwargs['grid']

        obs_x = []
        obs_y = []
        while len(obs_x) < nobs:
            y = np.random.uniform(grid.ymin, grid.ymax)
            x = np.random.uniform(grid.xmin, grid.xmax)
            valid = grid.interp(grid.mask.astype(int), x, y)
            if valid[0] == 0:
                obs_x.append(x)
                obs_y.append(y)

        obs_x = np.array(obs_x)
        obs_y = np.array(obs_y)
        obs_z = np.zeros(nobs)

        obs_seq = {'obs': np.full(nobs, np.nan),
                't': np.full(nobs, kwargs['time']),
                'z': obs_z,
                'y': obs_y,
                'x': obs_x,
                'err_std': np.ones(nobs) * kwargs['err']['std']
                }
        return obs_seq

    def read_obs(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        grid = kwargs['grid']
        mask = kwargs['mask']
        name = kwargs['name']
        native_name = self.variables[name].name
        assert isinstance(native_name, str), f"{native_name} is invalid name"

        obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[], }

        flist = self.filename(**kwargs)
        # CS2SMOS data is only available during winter (defined by self.season_start/end*)
        #if filename returns empty list, then just return an empty obs_seq
        if len(flist) == 0:
            obs_seq_arr = {}
            for key in obs_seq.keys():
                obs_seq_arr[key] = np.array(obs_seq[key])
            return obs_seq_arr

        for fname in flist:
            #read the data file
            f = netCDF4.Dataset(fname, 'r')

            lat = f['lat'][...].data.flatten()
            lon = f['lon'][...].data.flatten()
            x_, y_ = grid.proj(lon, lat)
            mask_ = grid.interp(mask.astype(int), x_, y_)

            ntime = f.dimensions['time'].size
            for n in range(ntime):
                t = f['time'][n].data * timedelta(seconds=1) + datetime(1978, 1, 1, tzinfo=timezone.utc)
                obs = f[native_name][n,...].data.flatten()

                if 'analysis_sea_ice_thickness_unc' in f.variables:
                    obs_err = f['analysis_sea_ice_thickness_unc'][n,...].data.flatten()
                else:
                    obs_err = 0.1 * np.ones(obs.shape)  # Default error if not available

                for p in range(obs.size):
                    if mask_[p] > 0:
                        continue
                    if x_[p] < grid.xmin or x_[p] > grid.xmax or y_[p] < grid.ymin or y_[p] > grid.ymax:
                        continue

                    obs_value = obs[p]

                    if self.use_dataset_uncertainty:
                        obs_err_std = obs_err[p]  # use uncertainty from dataset
                    else:
                        obs_err_std = kwargs['err']['std']

                    #assign to obs_seq
                    obs_seq['obs'].append(obs_value)
                    obs_seq['err_std'].append(obs_err_std)
                    obs_seq['t'].append(t)
                    obs_seq['z'].append(0)
                    obs_seq['y'].append(y_[p])
                    obs_seq['x'].append(x_[p])

            f.close()

        obs_seq_arr = {}
        for key in obs_seq.keys():
            obs_seq_arr[key] = np.array(obs_seq[key])
        return obs_seq_arr
