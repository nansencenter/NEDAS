import os
import glob
import numpy as np
from datetime import datetime, timedelta, timezone
from NEDAS.utils.conversion import dt1h
from NEDAS.datasets import Dataset

class ArgoObs(Dataset):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.variables = {
            'ocean_temp':{'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'K'},
            'ocean_saln':{'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'psu'},
            }
        self.obs_operator = {}

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        path = kwargs['path']
        time = kwargs['time']
        obs_window_min = kwargs['obs_window_min']
        obs_window_max = kwargs['obs_window_max']

        if time is None:
            tstr = '????????'
            search = os.path.join(path, tstr+'_?_prof.nc')
            file_list = glob.glob(search)

        else:
            if obs_window_min is not None and obs_window_max is not None:
                d_range = np.arange(obs_window_min, obs_window_max)
            else:
                d_range = [0]

            file_list = []
            for d in d_range:
                t = time + d * dt1h
                tstr = t.strftime('%Y%m%d')
                search = os.path.join(path, tstr+'_?_prof.nc')
                for result in glob.glob(search):
                    if result not in file_list:
                        file_list.append(result)

        assert len(file_list)>0, 'no matching files found: '+search
        return file_list

    def read_obs(self, **kwargs):
        """
        read raw data file, quality check, thinning according to model grid, then output to obs_seq
        kwargs:
        -grid: model 2D grid [ny, nx]
        -mask: masked or not (for land or other reason) bool[ny, nx]
        -z: depth (-meters) defined on grid [nz, ny, nx], if None not available
        -name, time, etc.
        return: obs_seq dict with lists of obs,x,y,z,time
        """
        kwargs = super().parse_kwargs(**kwargs)
        name = kwargs['name']
        
        ##target grid for obs_seq
        grid = kwargs['grid']
        mask = kwargs['mask']

        model = kwargs['model']
        assert model is not None, 'read_obs: ERROR: model is required for argo dataset'
        model.grid.set_destination_grid(grid)
      
        obs_seq = {'obs':[],
                   't':[], 'z':[], 'y':[], 'x':[],
                   'err_std':[],
                   'profile_id':[], 'level_id':[] }

        for fname in self.filename(**kwargs):

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

            ##find z coordinate in meters, negative relative to surface
            if 'PRES' in f.variables:
                z = 0. - f['PRES'][0:nprof, 0:nlev].data * 1e4 / self.ONEM ##decibar to meter
                z_qc = f['PRES_QC'][0:nprof, 0:nlev].data
            elif 'DEPH' in f.variables:
                z = 0. - f['DEPH'][0:nprof, 0:nlev].data
                z_qc = f['DEPH_QC'][0:nprof, 0:nlev].data

            ##model z (zm) at profile location on grid
            if model.z is not None:
                nz = model.z.shape[0]
                zm = np.zeros((nprof, nz))
                for k in range(nz):
                    ztmp = model.grid.convert(model.z[k, ...])
                    zm[:, k] = grid.interp(ztmp, x, y)

            ##observed variable
            if name == 'ocean_temp' and 'TEMP' in f.variables:
                obs = f['TEMP'][0:nprof, 0:nlev].data
                obs_qc = f['TEMP_QC'][0:nprof, 0:nlev].data
            elif name == 'ocean_saln' and 'PSAL' in f.variables:
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
                    if obs_qc[p,l] not in (b'1',):
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
                if model.z is not None:
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
                    if name == 'ocean_temp':
                        if obs[p,l] < self.TEMP_MIN or obs[p,l] > self.TEMP_MAX:
                            flag1[p] = 0
                            flag2[p,:] = 0
                    if name == 'ocean_saln':
                        if obs[p,l] < self.SALN_MIN or obs[p,l] > self.SALN_MAX:
                            flag1[p] = 0
                            flag2[p,:] = 0

            ##TODO: thinning in horizontal

            ##thinning in vertical
            if self.NUM_OBS_PER_LAYER is not None and model.z is not None:
                nz = model.z.shape[0]
                for p in range(nprof):
                    if flag1[p] == 0:
                        continue
                    ##target z levels to bin profile obs
                    zt = np.interp(np.arange(0, nz, 1/self.NUM_OBS_PER_LAYER),
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
                t_prof = datetime(1950,1,1,tzinfo=timezone.utc) + juld[p]*timedelta(days=1)

                for l in range(nlev):
                    if flag2[p,l] == 0:
                        continue
                    obs_seq['obs'].append(obs[p,l])
                    obs_seq['t'].append(t_prof)
                    obs_seq['z'].append(z[p,l])
                    obs_seq['y'].append(y[p])
                    obs_seq['x'].append(x[p])
                    obs_seq['err_std'].append(np.sqrt(self.OBS_ERR_VAR[name]))
                    obs_seq['profile_id'].append(p)
                    obs_seq['level_id'].append(l)

            f.close()

        for key in obs_seq.keys():
            obs_seq[key] = np.array(obs_seq[key])

        return obs_seq

    def random_network(self, model, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        raise NotImplementedError
        obs_seq = {}
        return obs_seq
