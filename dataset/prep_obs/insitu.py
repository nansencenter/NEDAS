import numpy as np
import importlib
import glob
from datetime import datetime, timedelta
from netCDF4 import Dataset

variables = {'ocean_temp':{'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
             'ocean_saln':{'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'psu'},
            }


def filename(path, **kwargs):
    if 'time' in kwargs:
        t = kwargs['time']
        tstr = t.strftime('%Y%m%d')
    else:
        tstr = '????????'

    search = path+'/'+tstr+'_?_prof.nc'
    flist = glob.glob(search)

    assert len(flist)>0, 'no matching files found: '+search

    return flist


##
## inputs, grid: model 2D grid [ny, nx]
##         mask: masked or not (for land or other reason) bool[ny, nx]
##         model_z: depth (-meters) defined on grid [nz, ny, nx]
##
def read_obs(path, grid, mask, model_z, **kwargs):
    nz, ny, nx = model_z.shape
    assert 'name' in kwargs, 'prep_obs.insitu.read_obs: missing obs variable name=?'
    obsname = kwargs['name']

    obs_seq = {'obs':[], 'x':[], 'y':[], 'z':[], 'time':[]}

    ONEM = 9806.           ##press (Pa) for 1 m depth
    NUM_OBS_PER_LAYER = 3  ##for vertical superobing
    TEMP_MIN = -2.         ##min,max of ocean_temp
    TEMP_MAX = 40.
    SALN_MIN = 16.         ##min,max of ocean_saln
    SALN_MAX = 37.5

    for fname in filename(path, **kwargs):

        ##read the profiles from nc file
        f = Dataset(fname)

        nprof = f.dimensions['N_PROF'].size
        nlev = f.dimensions['N_LEVELS'].size

        lat = f['LATITUDE'][0:nprof].data
        lon = f['LONGITUDE'][0:nprof].data
        x, y = grid.proj(lon, lat)  ##coordinates in grid

        juld = f['JULD'][0:nprof].data
        juld_qc = f['JULD_QC'][0:nprof].data
        pos_qc = f['POSITION_QC'][0:nprof].data

        ##DEPH and PRES both provide z coordinate in meters (approximately)
        if 'PRES' in f.variables:
            z = f['PRES'][0:nprof, 0:nlev].data *1e4/ONEM ##decibar to meter
            z_qc = f['PRES_QC'][0:nprof, 0:nlev].data
        elif 'DEPH' in f.variables:
            z = f['DEPH'][0:nprof, 0:nlev].data
            z_qc = f['DEPH_QC'][0:nprof, 0:nlev].data
        ##flip sign for ocean depth is negative relative to surface
        z = -z

        ##model z at profile location
        zm = np.zeros((nprof, model_z.shape[0]))
        for k in range(model_z.shape[0]):
            zm[:, k] = grid.interp(model_z[k,...], x, y)

        ##observed variable
        if obsname == 'ocean_temp' and 'TEMP' in f.variables:
            obs = f['TEMP'][0:nprof, 0:nlev].data
            obs_qc = f['TEMP_QC'][0:nprof, 0:nlev].data
        elif obsname == 'ocean_saln' and 'PSAL' in f.variables:
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
        for p in range(nprof):
            if flag1[p] == 0:
                continue
            ##check if profile location is masked in model grid
            if grid.interp(mask.astype(int), x[p], y[p])>0:
                flag1[p] = 0
                continue
            ##check if z level is deeper than model depth at profile location
            for l in range(nlev):
                if z[p,l] < zm[p,-1]:
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
                if obsname == 'ocean_temp':
                    if obs[p,l] < TEMP_MIN or obs[p,l] > TEMP_MAX:
                        flag1[p] = 0
                        flag2[p,:] = 0
                if obsname == 'ocean_saln':
                    if obs[p,l] < SALN_MIN or obs[p,l] > SALN_MAX:
                        flag1[p] = 0
                        flag2[p,:] = 0

        ##6. thinning/superobing in vertical
        for p in range(nprof):
            if flag1[p] == 0:
                continue

            ##target z levels to thin obs
            zt = np.interp(np.arange(0, nz, 1/NUM_OBS_PER_LAYER),
                           np.arange(1, nz+1),
                           zm[p, :])
            zt_ind = []
            for l in range(nlev):
                if flag2[p,l] == 0:
                    continue
                ind = np.searchsorted(zt[::-1], z[p, l])
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
                obs_seq['time'].append(t_prof)
                obs_seq['x'].append(x[p])
                obs_seq['y'].append(y[p])
                obs_seq['z'].append(z[p,l])
                obs_seq['obs'].append(obs[p,l])

    return obs_seq


def obs_operator():
    pass



