import numpy as np
import glob
from datetime import datetime, timedelta
import pyproj
from netCDF4 import Dataset
from grid import Grid

variables = {'seaice_drift': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'km/day'}, }

##osisaf grid definition
proj = pyproj.Proj("+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45")
x, y = np.meshgrid(np.arange(-3750000, 3626000, 62500), np.arange(5750000, -5251000, -62500))
obs_grid = Grid(proj, x, y)


def filename(path, **kwargs):
    if 'time' not in kwargs:
        search = path+'/????/??/ice_drift_nh_polstere-625_multi-oi_????????????-????????????.nc'
        file_list = glob.glob(search)

    else:
        if 'obs_window_min' in kwargs and 'obs_window_max' in kwargs:
            d_range = np.arange(kwargs['obs_window_min'], kwargs['obs_window_max'])
        else:
            d_range = [0]

        file_list = []
        for d in d_range:
            t = kwargs['time'] + d * timedelta(hours=1)
            t0 = t - timedelta(days=2)  ##drift traj start time
            search = path+'/'+t.strftime('%Y')+'/'+t.strftime('%m')+'/ice_drift_nh_polstere-625_multi-oi_'+t0.strftime('%Y%m%d%H%M')+'-'+t.strftime('%Y%m%d%H%M')+'.nc'
            for result in glob.glob(search):
                if result not in file_list:
                    file_list.append(result)

    assert len(file_list)>0, 'no matching files found: '+search

    return file_list


def read_obs(path, grid, mask, model_z, **kwargs):
    assert 'name' in kwargs, 'osisaf.ice_conc.read_obs: missing obs variable name'
    assert kwargs['name'] == 'seaice_drift', 'osisaf.ice_conc.read_obs can only provide seaice_conc'

    obs_grid.set_destination_grid(grid)  ##for vector rotation

    obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[], }

    for file_name in filename(path, **kwargs):

        ##read the data file
        f = Dataset(file_name)

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
            obs_drift = obs_grid.rotate_vectors(np.array([obs_dx, obs_dy]))
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
                obs_t = (t1+dt1) * timedelta(seconds=1) + datetime(1978, 1, 1)
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


