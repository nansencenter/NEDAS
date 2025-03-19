import os
import glob
import numpy as np
from datetime import datetime, timedelta
import pyproj
from scipy.spatial import KDTree
from .utils import get_data_traj_pairs, get_triangulation, get_velocity, get_velocity_gradients
from ..dataset_config import DatasetConfig

class Dataset(DatasetConfig):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        ##variable dictionary for RGPS naming convention
        self.variables = {
            'seaice_drift': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'km/day'},
            'seaice_shear': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
            'seaice_div': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
            'seaice_vort': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
            }

        ##RGPS trajectory data (x,y) is in NorthPolarStereo projection:
        self.proj = pyproj.Proj(self.proj4)

        ##tolerance when searching for the time along trajectory
        self.dt_tol=timedelta(days=self.DAYS_SEARCH_TOLERANCE)

        self.obs_operator = {
            'seaice_drift': self.get_seaice_property('seaice_drift'),
            'seaice_shear': self.get_seaice_property('seaice_shear'),
            'seaice_div': self.get_seaice_property('seaice_div'),
            'seaice_vort': self.get_seaice_property('seaice_vort'),
            }

        def parse_kwargs(self, **kwargs):
            kwargs = super().parse_kwargs(**kwargs)
            assert kwargs['time'] is not None, 'rgps.read_obs: time is not defined'
            assert kwargs['obs_window_max'] is not None, 'rgps.read_obs: obs_window_max is not defined'
            assert kwargs['obs_window_min'] is not None, 'rgps.read_obs: obs_window_min is not defined'
            return kwargs

        def filename(self, **kwargs):
            kwargs = parse_kwargs(**kwargs)
            t = kwargs['time']

            if 'obs_window_min' in kwargs and 'obs_window_max' in kwargs:
                d_range = [kwargs['obs_window_min'], kwargs['obs_window_max']]
            else:
                d_range = [0]

            file_list = []
            search = os.path.join(kwargs['path'], 'RGPS*traj.mat')
            for result in glob.glob(search):
                ss = result.split('_')
                t1 = datetime.strptime(ss[1], '%Y-%m-%d')
                t2 = datetime.strptime(ss[2], '%Y-%m-%d')
                for d in d_range:
                    t_ = t + d * timedelta(hours=1)
                    if t_ >= t1 and t_ <= t2 and result not in file_list:
                        file_list.append(result)

            assert len(file_list)>0, 'no matching files found'

            return file_list

        def read_obs(self, **kwargs):
            """read obs from rgps dataset"""
            kwargs = parse_kwargs(**kwargs)
            obs_name = kwargs['name']
            d0_out = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_min']
            d1_out = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_max']

            ##target grid for obs_seq
            grid = kwargs['grid']

            ##note: x,y are obs location on grid.proj (in meters)
            ##      x0,y0,x1,y1 are position on rgps_proj (in kilometers),
            ##      triangles: velocity is defined on nodes and deform on elements
            ##      record: trajectory id, we process one record at a time
            obs_seq = {'obs':[], 'err_std':[], 't':[], 'z':[], 'y':[], 'x':[],
                       'x0':[], 'y0':[], 'triangles':[], 'record':[], 'index':[]}

            rec = 0
            for file_name in self.filename(**kwargs):
                pairs = get_data_traj_pairs(file_name, d0_out, d1_out, self.dt_tol)

                for x0, y0, t0, x1, y1, t1 in pairs:
                    tri = get_triangulation(x0, y0)
                    u, v = get_velocity(x0, y0, t0, x1, y1, t1)

                    ##convert x0,y0 from rgps_proj to grid.proj in meters
                    x, y = grid.proj(*self.proj(x0*1000, y0*1000, inverse=True))

                    if obs_name == 'seaice_drift':
                        for p in range(len(x)):
                            ##quality check
                            if np.hypot(u[p], v[p]) > self.DRIFT_MAX or np.isnan(u[p]) or np.isnan(v[p]):
                                continue

                            ##add the obs to sequence
                            obs_seq['obs'].append([u[p], v[p]])
                            obs_seq['err_std'].append(self.DRIFT_ERR_STD)
                            obs_seq['t'].append(t0[p])
                            obs_seq['z'].append(0.)
                            obs_seq['y'].append(y[p])
                            obs_seq['x'].append(x[p])
                            obs_seq['record'].append(rec)
                            obs_seq['index'].append(p)

                    elif obs_name in ['seaice_shear', 'seaice_div', 'seaice_vort']:
                        x_elem = np.mean(x[tri.triangles], axis=1)
                        y_elem = np.mean(y[tri.triangles], axis=1)
                        t_elem = t0[tri.triangles[:, 0]]
                        ux, uy, vx, vy = self.get_velocity_gradients(x0, y0, u, v)

                        if obs_name == 'seaice_shear':
                            obs_values = np.hypot(ux - vy, uy + vx)
                        elif obs_name == 'seaice_div':
                            obs_values = ux + vy
                        elif obs_name == 'seaice_vort':
                            obs_values = vx - uy

                        for p in range(len(x_elem)):
                            ##quality check
                            if tri.mask[p]:
                                continue
                            if np.abs(obs_values[p]) > self.DEFORM_MAX or np.isnan(obs_values[p]):
                                continue

                            ##add the obs to sequence
                            obs_seq['obs'].append(obs_values[p])
                            obs_seq['err_std'].append(self.DEFORM_ERR_STD)
                            obs_seq['t'].append(t_elem[p])
                            obs_seq['z'].append(0.)
                            obs_seq['y'].append(y_elem[p])
                            obs_seq['x'].append(x_elem[p])
                            obs_seq['record'].append(rec)
                            obs_seq['index'].append(p)

                    ##save raw data for state_to_obs to use
                    obs_seq['x0'].append(x0)
                    obs_seq['y0'].append(y0)
                    obs_seq['triangles'].append(tri.triangles)
                    rec += 1

            ##convert from list to np.array
            ##raw data are kept in list format
            for key in ('obs', 'err_std', 't', 'y', 'x', 'z', 'record', 'index'):
                obs_seq[key] = np.array(obs_seq[key])

            if self.variables[obs_name]['is_vector']:
                obs_seq['obs'] = obs_seq['obs'].T

            return obs_seq

        def random_network(self, **kwargs):
            raise NotImplementedError('rgps.random_network: random_network not implemented for rgps dataset')

        def get_model_files(self, **kwargs):
            kwargs = parse_kwargs(**kwargs)

            ##start and end time of the trajectories
            t0 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_min']
            t1 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_max']

            if kwargs['member'] is not None:
                mstr = '{:03d}'.format(kwargs['member']+1)
            else:
                mstr = ''

            file_list = []
            t_list = []
            search = os.path.join(kwargs['path'], mstr, '/mesh*.bin')
            for result in glob.glob(search):
                if '00Z' not in result:  ##discard files without time string
                    continue
                tstr = result.split('.')[-2].split('_')[-1]
                if 'post_regrid' in result:
                    tstr = tstr.replace('0Z', '1Z')
                t = datetime.strptime(tstr, '%Y%m%dT%H%M%SZ')
                if t<t0 or t>t1:
                    continue
                file_list.append(result)
                t_list.append(t)

            file_list = list(np.array(file_list)[np.argsort(t_list)])
            assert len(file_list)>0, 'no matching files found'

            return file_list

        def get_traj_pairs(self, **kwargs):
            """get nextsim simulated trajectory pairs, corresponding to rgps"""
            kwargs = parse_kwargs(**kwargs)
            model = kwargs['model']

            ##rgps start positions on rgps_proj, in km units
            x0, y0 = kwargs['x0'], kwargs['y0']

            ##make a copy of start position to be updated to final position
            nrec = len(x0)
            x, y, i, dx, dy = [], [], [], [], []
            for r in range(nrec):
                x.append(x0[r].copy())
                y.append(y0[r].copy())
                i.append(np.zeros(x0[r].shape, dtype='int'))
                dx.append(np.zeros(x0[r].shape))
                dy.append(np.zeros(x0[r].shape))

            ##time for the start/end position
            t0 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_min']
            t1 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_max']

            file_list = get_model_files(**kwargs)

            for n in range(len(file_list)):
                fname = file_list[n]

                ##get model mesh points in native proj, in meters
                model.read_grid(**kwargs)
                mx = model.grid.x
                my = model.grid.y
                ##convert to rgps_proj, in km units
                x_, y_ = self.proj(*model.grid.proj(mx, my, inverse=True))
                mx, my = x_/1000, y_/1000

                if n==0 or 'post_regrid' in fname:
                    ##build kdtree for spatial search if model mesh changed
                    tree = KDTree(np.vstack([mx, my]).T)

                    ##search for traj position in mesh
                    for r in range(nrec):
                        d, i[r] = tree.query(np.vstack([x[r], y[r]]).T)
                        dx[r], dy[r] = x[r] - mx[i[r]], y[r] - my[i[r]]

                if 'post_regrid' not in fname:
                    ##move to next position
                    for r in range(nrec):
                        x[r], y[r] = mx[i[r]] + dx[r], my[i[r]] + dy[r]

            pairs = []
            for r in range(nrec):
                pairs.append([x0[r], y0[r], np.full(x0[r].shape, t0),
                            x[r],  y[r], np.full(x0[r].shape, t1)])
            return pairs

        def get_seaice_property(self, obs_name):
            def operator(self, **kwargs):
                kwargs = super().parse_kwargs(**kwargs)
                obs_name = kwargs['name']

                pairs = self.get_traj_pairs(**kwargs)

                xo, yo, tri, r, i = kwargs['x'], kwargs['y'], kwargs['triangles'], kwargs['record'], kwargs['index']
                nobs = xo.size

                if self.variables[obs_name]['is_vector']:
                    obs_seq = np.full((2, nobs), np.nan)
                else:
                    obs_seq = np.full(nobs, np.nan)

                rec = 0
                for x0, y0, t0, x1, y1, t1 in pairs:
                    rind = np.where(r==rec)  ##indices in xo for record rec
                    ind = i[rind]            ##indices in x0 that forms the record rec part of xo

                    u, v = get_velocity(x0, y0, t0, x1, y1, t1)
                    ux, uy, vx, vy = get_velocity_gradients(x0, y0, u, v)

                    if obs_name == 'seaice_drift':
                        obs_seq[0, rind] = u[ind]
                        obs_seq[1, rind] = v[ind]

                    elif obs_name == 'seaice_shear':
                        obs_values = np.hypot(ux - vy, uy + vx)
                        obs_seq[rind] = obs_values[ind]

                    elif obs_name == 'seaice_div':
                        obs_values = ux + vy
                        obs_seq[rind] = obs_values[ind]

                    elif obs_name == 'seaice_vort':
                        obs_values = vx - uy
                        obs_seq[rind] = obs_values[ind]

                    rec += 1
                return obs_seq
            return operator
