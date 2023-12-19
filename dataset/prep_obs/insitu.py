import numpy as np
import importlib
import glob
from datetime import datetime, timedelta
from netCDF4 import Dataset

variables = {'ocean_temp':{'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
             'ocean_saln':{'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'psu'},
            }


def filename(path, **kwargs):
    if 'time' not in kwargs:
        tstr = '????????'
        search = path+'/'+tstr+'_?_prof.nc'
        file_list = glob.glob(search)

    else:
        if 'obs_window_min' in kwargs and 'obs_window_max' in kwargs:
            d_range = np.arange(kwargs['obs_window_min'], kwargs['obs_window_max'])
        else:
            d_range = [0]

        file_list = []
        for d in d_range:
            t = kwargs['time'] + d * timedelta(hours=1)
            tstr = t.strftime('%Y%m%d')
            search = path+'/'+tstr+'_?_prof.nc'
            for result in glob.glob(search):
                if result not in file_list:
                    file_list.append(result)

    assert len(file_list)>0, 'no matching files found: '+search

    return file_list


##read raw data file, quality check, thinning then output to obs_seq
## inputs, grid: model 2D grid [ny, nx]
##         mask: masked or not (for land or other reason) bool[ny, nx]
##         model_z: depth (-meters) defined on grid [nz, ny, nx], if None not available
## return: obs_seq dict with lists of obs,x,y,z,time
def read_obs(path, grid, mask, model_z, **kwargs):
    assert 'name' in kwargs, 'prep_obs.insitu.read_obs: missing obs variable name=?'
    obs_name = kwargs['name']

    obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[]}

    ONEM = 9806.           ##press (Pa) for 1 m depth
    NUM_OBS_PER_LAYER = 1  ##for vertical superobing
    TEMP_MIN = -2.         ##min,max of ocean_temp
    TEMP_MAX = 40.
    SALN_MIN = 16.         ##min,max of ocean_saln
    SALN_MAX = 37.5

    for file_name in filename(path, **kwargs):

        ##read the profiles from nc file
        f = Dataset(file_name)

        nprof = f.dimensions['N_PROF'].size
        nlev = f.dimensions['N_LEVELS'].size

        lat = f['LATITUDE'][0:nprof].data
        lon = f['LONGITUDE'][0:nprof].data
        x, y = grid.proj(lon, lat)  ##coordinates in grid

        juld = f['JULD'][0:nprof].data
        juld_qc = f['JULD_QC'][0:nprof].data
        pos_qc = f['POSITION_QC'][0:nprof].data

        ##find z coordinate in meters, negative relative to surface
        if 'PRES' in f.variables:
            z = 0. - f['PRES'][0:nprof, 0:nlev].data * 1e4 / ONEM ##decibar to meter
            z_qc = f['PRES_QC'][0:nprof, 0:nlev].data
        elif 'DEPH' in f.variables:
            z = 0. - f['DEPH'][0:nprof, 0:nlev].data
            z_qc = f['DEPH_QC'][0:nprof, 0:nlev].data

        ##model z (zm) at profile location
        if model_z is not None:
            nz, ny, nx = model_z.shape
            zm = np.zeros((nprof, nz))
            for k in range(nz):
                zm[:, k] = grid.interp(model_z[k, ...], x, y)

        ##observed variable
        if obs_name == 'ocean_temp' and 'TEMP' in f.variables:
            obs = f['TEMP'][0:nprof, 0:nlev].data
            obs_qc = f['TEMP_QC'][0:nprof, 0:nlev].data
        elif obs_name == 'ocean_saln' and 'PSAL' in f.variables:
            obs = f['PSAL'][0:nprof, 0:nlev].data
            obs_qc = f['PSAL_QC'][0:nprof, 0:nlev].data
        else:
            continue

        ##quality check flags
        flag1 = np.ones(nprof)
        flag2 = np.ones((nprof, nlev))

        for p in range(nprof):
            if juld_qc[p] not in (b'1', b'2'):
                flag1[p] = 0
            if pos_qc[p] not in (b'1', b'2'):
                flag1[p] = 0

        ##check inside grid
        for p in range(nprof):
            if x[p]<grid.xmin or x[p]>grid.xmax or y[p]<grid.ymin or y[p]>grid.ymax:
                flag1[p] = 0

        for p in range(nprof):
            for l in range(nlev):
                if z_qc[p,l] not in (b'1', b'2'):
                    flag2[p,l] = 0
                if obs_qc[p,l] not in (b'1', b'2'):
                    flag2[p,l] = 0
            if np.sum(flag2[p,:]) == 0:
                flag1[p] = 0

        ##check if location is unmasked (wet)
        mask_prof = grid.interp(mask.astype(int), x, y)
        for p in range(nprof):
            if flag1[p] == 0:
                continue
            ##check if profile location is masked in model grid
            if mask_prof[p] > 0:
                flag1[p] = 0
                continue
            ##check if z level is deeper than model depth at profile location
            if model_z is not None:
                for l in range(nlev):
                    if z[p,l] < np.min(zm[p,:]):
                        flag2[p,l] = 0
                if np.sum(flag2[p,:]) == 0:
                    flag1[p] = 0

        ##check physical value
        for p in range(nprof):
            if flag1[p] == 0:
                continue
            for l in range(nlev):
                if flag2[p,l] == 0:
                    continue
                if obs_name == 'ocean_temp':
                    if obs[p,l] < TEMP_MIN or obs[p,l] > TEMP_MAX:
                        flag1[p] = 0
                        flag2[p,:] = 0
                if obs_name == 'ocean_saln':
                    if obs[p,l] < SALN_MIN or obs[p,l] > SALN_MAX:
                        flag1[p] = 0
                        flag2[p,:] = 0

        ##thinning in horizontal TODO


        ##thinning in vertical
        if NUM_OBS_PER_LAYER is not None and model_z is not None:
            nz, ny, nx = model_z.shape
            for p in range(nprof):
                if flag1[p] == 0:
                    continue
                ##target z levels to bin profile obs
                zt = np.interp(np.arange(0, nz, 1/NUM_OBS_PER_LAYER),
                               np.arange(1, nz+1, 1), zm[p, :])
                zt = np.sort(zt)
                zt_ind = []
                for l in range(nlev):
                    if flag2[p,l] == 0:
                        continue
                    ind = np.searchsorted(zt, z[p, l])
                    if ind not in zt_ind:
                        zt_ind.append(ind)
                    else:
                        flag2[p,l] = 0

        ##check convective stability is not implemented here
        ##  potential_density(temp, saln), find vertical increment from level k to k+1
        ##  if density difference rho_inc < DENS_DIFF_MIN = -0.02
        ##  then discard this profile

        ##output data to the obs_seq
        for p in range(nprof):
            if flag1[p] == 0:
                continue

            ##time of the profile given by julian day
            t_prof = datetime(1950,1,1) + juld[p]*timedelta(days=1)

            for l in range(nlev):
                if flag2[p,l] == 0:
                    continue
                obs_seq['obs'].append(obs[p,l])
                obs_seq['t'].append(t_prof)
                obs_seq['z'].append(z[p,l])
                obs_seq['y'].append(y[p])
                obs_seq['x'].append(x[p])

    for key in obs_seq.keys():
        obs_seq[key] = np.array(obs_seq[key])

    return obs_seq


##
def random_network():
    return obs_seq


##
obs_operator = {}


