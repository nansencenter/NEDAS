import numpy as np
import os
import glob
from datetime import datetime, timedelta, timezone
import netCDF4
import pyproj
from NEDAS.grid import Grid
from NEDAS.datasets import Dataset

class OsisafSeaIceConcObs(Dataset):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {'seaice_conc': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':1}, }

        proj = pyproj.Proj(self.proj)
        x, y = np.meshgrid(np.arange(self.xstart, self.xend, self.dx), np.arange(self.ystart, self.yend, self.dy))
        self.grid = Grid(proj, x, y)

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        path = kwargs['path']
        time = kwargs['time']
        name = kwargs['name']
        obs_window_min = kwargs['obs_window_min']
        obs_window_max = kwargs['obs_window_max']

        if time is None:
            search = os.path.join(path, '????_'+self.proj_name, 'ice_conc_'+self.proj_name+'-100_multi_????????????.nc')
            file_list = glob.glob(search)

        else:
            if obs_window_min is not None and obs_window_max is not None:
                d_range = np.arange(obs_window_min, obs_window_max)
            else:
                d_range = [0]

            file_list = []
            for d in d_range:
                t = time + d * timedelta(hours=1)
                tstr = t.strftime('%Y%m%d%H%M')
                search = os.path.join(path, t.strftime('%Y')+'_'+self.proj_name, 'ice_conc_'+self.proj_name+'-100_multi_'+tstr+'.nc')
                for result in glob.glob(search):
                    if result not in file_list:
                        file_list.append(result)

        assert len(file_list)>0, 'no matching files found: '+search
        return file_list

    def random_network(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

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
            if valid == 0:
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
        kwargs = super().parse_kwargs(**kwargs)
        grid = kwargs['grid']
        mask = kwargs['mask']

        obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[], }

        for fname in self.filename(**kwargs):

            ##read the data file
            f = netCDF4.Dataset(fname)

            lat = f['lat'][...].data.flatten()
            lon = f['lon'][...].data.flatten()
            x_, y_ = grid.proj(lon, lat)
            mask_ = grid.interp(mask.astype(int), x_, y_)

            ntime = f.dimensions['time'].size
            for n in range(ntime):
                t = f['time'][n].data * timedelta(seconds=1) + datetime(1978, 1, 1, tzinfo=timezone.utc)
                qc_flag = f['status_flag'][n,...].data.flatten()
                obs = f['ice_conc'][n,...].data.flatten()
                if 'total_uncertainty' in f.variables:
                    obs_err = f['total_uncertainty'][n,...].data.flatten()
                else:
                    obs_err = 0.3 * np.ones(obs.shape)  ##default conc err 0.3

                for p in range(obs.size):
                    if qc_flag[p] < 0 or qc_flag[p] > 10:
                        continue
                    if obs_err[p] <= 0:
                        continue

                    if x_[p] < grid.xmin or x_[p] > grid.xmax or y_[p] < grid.ymin or y_[p] > grid.ymax:
                        continue

                    if mask_[p] > 0:
                        continue

                    obs_value = obs[p] * 0.01   ##convert percent to 0-1

                    #obs_err_var = (obs_err[p]*0.01)**2  ##uncertainty from dataset
                    obs_err_var = 1. + (0.5 - np.abs(0.5-obs_value))**2  ##adaptive error used in topaz

                    ##assignn to obs_seq
                    obs_seq['obs'].append(obs_value)
                    obs_seq['err_std'].append(np.sqrt(obs_err_var))
                    obs_seq['t'].append(t)
                    obs_seq['z'].append(0)
                    obs_seq['y'].append(y_[p])
                    obs_seq['x'].append(x_[p])

            f.close()

        for key in obs_seq.keys():
            obs_seq[key] = np.array(obs_seq[key])

        return obs_seq

