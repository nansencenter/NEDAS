import numpy as np
import os
import glob
from datetime import datetime, timedelta, timezone
import netCDF4
import pyproj
from NEDAS.grid import Grid
from NEDAS.datasets import Dataset

class OsisafSeaIceDriftObs(Dataset):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {'seaice_drift': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'km/day'}, }

        proj = pyproj.Proj(self.proj)
        x, y = np.meshgrid(np.arange(self.xstart, self.xend, self.dx), np.arange(self.ystart, self.yend, self.dy))
        self.grid = Grid(proj, x, y)

        self.obs_operator = {'seaice_drift': self.get_seaice_drift,}

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        path = kwargs['path']
        time = kwargs['time']
        name = kwargs['name']
        obs_window_min = kwargs['obs_window_min']
        obs_window_max = kwargs['obs_window_max']

        if time is None:
            search = os.path.join(path, '????', '??', 'ice_drift_'+self.proj_name+'-625_multi-oi_????????????-????????????.nc')
            file_list = glob.glob(search)

        else:
            if obs_window_min is not None and obs_window_max is not None:
                d_range = np.arange(obs_window_min, obs_window_max)
            else:
                d_range = [0]

            file_list = []
            for d in d_range:
                t = time + d * timedelta(hours=1)
                t0 = t - timedelta(days=2)  ##drift traj start time
                search = os.path.join(path, t.strftime('%Y'), t.strftime('%m'), 'ice_drift_'+self.proj_name+'-625_multi-oi_'+t0.strftime('%Y%m%d%H%M')+'-'+t.strftime('%Y%m%d%H%M')+'.nc')
                for result in glob.glob(search):
                    if result not in file_list:
                        file_list.append(result)

        assert len(file_list)>0, 'no matching files found: '+search
        return file_list

    def read_obs(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        grid = kwargs['grid']
        mask = kwargs['mask']

        self.grid.set_destination_grid(grid)  ##for vector rotation

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
                t0, t1 = f['time_bnds'][n,:].data
                obs_dt0 = f['dt0'][n,...].data.flatten()
                obs_dt1 = f['dt1'][n,...].data.flatten()
                qc_flag = f['status_flag'][n,...].data.flatten()

                ###get drift vector, rotate vector from proj to grid.proj
                obs_dx = f['dX'][n,...]
                obs_dy = f['dY'][n,...]
                obs_drift = self.grid.rotate_vectors(np.array([obs_dx, obs_dy]))
                obs_dx = obs_drift[0,...].flatten()
                obs_dy = obs_drift[1,...].flatten()

                if 'uncert_dX_and_dY' in f.variables:
                    obs_err = f['uncert_dX_and_dY'][n,...].data.flatten()
                else:
                    obs_err = 10 * np.ones(obs_dx.shape)  ##default drift err 10 km

                for p in range(obs_dx.size):
                    if qc_flag[p] != 30:
                        continue
                    if obs_err[p] <= 0:
                        continue

                    if x_[p] < grid.xmin or x_[p] > grid.xmax or y_[p] < grid.ymin or y_[p] > grid.ymax:
                        continue

                    if mask_[p] > 0:
                        continue

                    dt0 = obs_dt0[p] if ~np.isnan(obs_dt0[p]) else 0.
                    dt1 = obs_dt1[p] if ~np.isnan(obs_dt1[p]) else 0.
                    obs_t = (t1+dt1) * timedelta(seconds=1) + datetime(1978, 1, 1, tzinfo=timezone.utc)
                    obs_dt = (t1+dt1 - t0-dt0)*timedelta(seconds=1) / timedelta(days=1)
                    obs_u = obs_dx[p] / obs_dt
                    obs_v = obs_dy[p] / obs_dt
                    obs_value = [obs_u, obs_v]

                    obs_err_std = obs_err[p] / obs_dt  ##uncertainty from dataset

                    ##assignn to obs_seq
                    obs_seq['obs'].append(obs_value)
                    obs_seq['err_std'].append(obs_err_std)
                    obs_seq['t'].append(obs_t)
                    obs_seq['z'].append(0)
                    obs_seq['y'].append(y_[p])
                    obs_seq['x'].append(x_[p])

            f.close()

        for key in obs_seq.keys():
            obs_seq[key] = np.array(obs_seq[key])

        ##make obs dimension [2,nobs] for vectors
        obs_seq['obs'] = obs_seq['obs'].T

        return obs_seq

    def get_seaice_drift(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        grid = kwargs['grid']
        path = kwargs['path']
        member = kwargs['member']
        model = kwargs['model']
        model.grid.set_destination_grid(grid)

        start_x = kwargs['x']
        start_y = kwargs['y']
        start_t = kwargs['t']
        x = start_x.copy()
        y = start_y.copy()
        obs_seq = np.zeros((2,)+x.shape)

        return obs_seq
