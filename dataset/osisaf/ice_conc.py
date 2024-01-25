import numpy as np
import glob
from datetime import datetime, timedelta
import pyproj
from netCDF4 import Dataset

variables = {'seaice_conc': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'%'}, }

##osisaf grid definition
proj = pyproj.Proj("+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45")
x, y = np.meshgrid(np.arange(-3845000, 3755000, 10000), np.arange(5845000, -5355000, -10000))

def filename(path, **kwargs):
    if 'time' not in kwargs:
        search = path+'/ice_conc_nh_polstere-100_multi_????????????.nc'
        file_list = glob.glob(search)

    else:
        if 'obs_window_min' in kwargs and 'obs_window_max' in kwargs:
            d_range = np.arange(kwargs['obs_window_min'], kwargs['obs_window_max'])
        else:
            d_range = [0]

        file_list = []
        for d in d_range:
            t = kwargs['time'] + d * timedelta(hours=1)
            tstr = t.strftime('%Y%m%d%H%M')
            search = path+'/ice_conc_nh_polstere-100_multi_'+tstr+'.nc'
            for result in glob.glob(search):
                if result not in file_list:
                    file_list.append(result)

    assert len(file_list)>0, 'no matching files found: '+search

    return file_list


def read_obs(path, grid, mask, model_z, **kwargs):
    assert 'name' in kwargs, 'osisaf.ice_conc.read_obs: missing obs variable name'
    assert kwargs['name'] == 'seaice_conc', 'osisaf.ice_conc.read_obs can only provide seaice_conc'

    obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[], }

    for file_name in filename(path, **kwargs):

        ##read the data file
        f = Dataset(file_name)

        lat = f['lat'][...].data.flatten()
        lon = f['lon'][...].data.flatten()

        ntime = f.dimensions['time'].size
        for n in range(ntime):
            t = f['time'][n].data * timedelta(seconds=1) + datetime(1978, 1, 1)
            qc_flag = f['status_flag'][n,...].data.flatten()
            obs = f['ice_conc'][n,...].data.flatten()
            obs_err = f['total_uncertainty'][n,...].data.flatten()

            for p in range(obs.size):
                if qc_flag[p] < 0 or qc_flag[p] > 10:
                    continue
                if obs_err[p] <= 0:
                    continue
                x, y = grid.proj(lon[p], lat[p])
                if x<grid.xmin or x>grid.xmax or y<grid.ymin or y>grid.ymax:
                    continue

                obs_value = obs[p] * 0.01   ##convert percent to 0-1

                #obs_err_var = (obs_err[p]*0.01)**2  ##uncertainty from dataset
                obs_err_var = 1. + (0.5 - np.abs(0.5-obs_value))**2  ##adaptive error used in topaz

                ##assignn to obs_seq
                obs_seq['obs'].append(obs_value)
                obs_seq['err_std'].append(np.sqrt(obs_err_var))
                obs_seq['t'].append(t)
                obs_seq['z'].append(0)
                obs_seq['y'].append(y)
                obs_seq['x'].append(x)

        f.close()

    for key in obs_seq.keys():
        obs_seq[key] = np.array(obs_seq[key])

    return obs_seq


