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

    search = datapath+'/'+ccyymmdd+'_?_prof.nc'
    flist = glob.glob(search)

    assert len(flist)>0, 'no matching files found'

    return flist


def read_obs(path, grid, **kwargs):
    assert 'name' in kwargs, 'prep_obs.insitu.read_obs: missing obs variable name=?'
    obsname = kwargs['name']

    obs = {'obs':[], 'x':[], 'y':[], 'z':[], 'time':[]}

    TEMP_MIN = -2.
    TEMP_MAX = 40.
    SALN_MIN = 16.
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
            z = f['PRES'][0:nprof, 0:nlev].data
            z_qc = f['PRES_QC'][0:nprof, 0:nlev].data
        elif 'DEPH' in f.variables:
            z = f['DEPH'][0:nprof, 0:nlev].data
            z_qc = f['DEPH_QC'][0:nprof, 0:nlev].data

        if obsname == 'ocean_temp' and 'TEMP' in f.variables:
            obs = f['TEMP'][0:nprof, 0:nlev].data
            obs_qc = f['TEMP_QC'][0:nprof, 0:nlev].data
        elif obsname == 'ocean_saln' and 'PSAL' in f.variables:
            obs = f['PSAL'][0:nprof, 0:nlev].data
            obs_qc = f['PSAL_QC'][0:nprof, 0:nlev].data
        else:
            continue

        flag1 = np.ones(nprof)
        flag2 = np.ones((nprof, nlev))

        ##1. check prof qc flags
        for p in range(nprof):
            if juld_qc[p] not in (b'1', b'2'):
                flag1[p] = 0
            if pos_qc[p] not in (b'1', b'2'):
                flag1[p] = 0

        ##2. check inside grid
        for p in range(nprof):
            if x[p]<grid.xmin or x[p]>grid.xmax or y[p]<grid.ymin or y[p]>grid.ymax:
                flag1[p] = 0

        ##3. data qc flag
        for p in range(nprof):
            for l in range(nlev):
                if z_qc[p,l] not in (b'1', b'2'):
                    flag2[p,l] = 0
                if obs_qc[p,l] not in (b'1', b'2'):
                    flag2[p,l] = 0
            if np.sum(flag2[p,:]) == 0:
                flag1[p] = 0

        ##4. check if location is unmasked (wet)
        for p in range(nprof):
            if flag1[p] == 0:
                continue
            ##check if profile location is masked in model grid
            if grid.interp(mask.astype(int), x[p], y[p])>0:
                flag1[p] = 0
                continue
            ##check if z level is within model grid at profile location
            ##TODO, take obs near model level only

        ##5. check physical value
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

        ##output
        for p in range(nprof):
            if flag1[p] == 0:
                continue
            for l in range(nlev):
                if flag2[p,l] == 0:
                    continue
                obs['time'].append(datetime(1950,1,1) + juld[p]*timedelta(days=1))
                obs['x'].append(x[p])
                obs['y'].append(y[p])
                obs['z'].append(z[p,l])
                obs['obs'].append(obs[p,l])

