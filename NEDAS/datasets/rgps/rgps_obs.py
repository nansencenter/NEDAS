import os
import glob
import numpy as np
from datetime import datetime, timedelta
import pyproj
from scipy.spatial import KDTree
from NEDAS.grid import Grid
from NEDAS.datasets import Dataset
from .utils import get_data_traj_pairs, get_triangulation, get_velocity, get_velocity_gradients, get_deform_div, get_deform_shear, get_deform_vort

class RgpsObs(Dataset):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        ##variable dictionary for RGPS naming convention
        self.variables = {
            'seaice_drift': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'km/day'},
            'seaice_deform_shear': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
            'seaice_deform_div': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
            'seaice_deform_vort': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
            }

        ##RGPS trajectory data (x,y) is in NorthPolarStereo projection:
        self.proj = pyproj.Proj(self.proj4)

        ##tolerance when searching for the time along trajectory
        self.dt_tol=timedelta(days=self.DAYS_SEARCH_TOLERANCE)

        self.obs_operator = {}
        ##TODO the get_traj_pairs functions sometimes fail, near the tree.query part, returns wrong index?
        ## for now use the model.diag_variables for deform variables instead
        #    'seaice_drift': self.get_seaice_drift,
        #    'seaice_deform_shear': self.get_seaice_deform_shear,
        #    'seaice_deform_div': self.get_seaice_deform_div,
        #    'seaice_deform_vort': self.get_seaice_deform_vort,
        #}

    def parse_kwargs(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        assert kwargs['time'] is not None, 'rgps.read_obs: time is not defined'
        assert kwargs['obs_window_max'] is not None, 'rgps.read_obs: obs_window_max is not defined'
        assert kwargs['obs_window_min'] is not None, 'rgps.read_obs: obs_window_min is not defined'
        return kwargs

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
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
        kwargs = super().parse_kwargs(**kwargs)
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

                elif obs_name in ['seaice_deform_shear', 'seaice_deform_div', 'seaice_deform_vort']:
                    x_elem = np.mean(x[tri.triangles], axis=1)
                    y_elem = np.mean(y[tri.triangles], axis=1)
                    t_elem = t0[tri.triangles[:, 0]]
                    ux, uy, vx, vy = get_velocity_gradients(x0, y0, u, v)

                    if obs_name == 'seaice_deform_shear':
                        obs_values = np.hypot(ux - vy, uy + vx)
                    elif obs_name == 'seaice_deform_div':
                        obs_values = ux + vy
                    elif obs_name == 'seaice_deform_vort':
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
        for key in ('obs', 'err_std', 't', 'y', 'x', 'z'):
            obs_seq[key] = np.array(obs_seq[key])

        if self.variables[obs_name]['is_vector']:
            obs_seq['obs'] = obs_seq['obs'].T

        ##superob to target grid, coarsen and keep the valid grid points
        ##make a mesh grid for the obs points
        obs_x, obs_y = obs_seq['x'], obs_seq['y']
        obs_grid = Grid(grid.proj, obs_x, obs_y, regular=False)
        ##remove unwanted triangles in the mesh
        msk = np.logical_or(obs_grid.tri.a > 2e8, obs_grid.tri.p > 1e5, obs_grid.tri.ratio < 0.3)
        obs_grid = Grid(grid.proj, obs_x, obs_y, regular=False, triangles=obs_grid.tri.triangles[~msk,:])
        ##convert to target grid with coarse graining (superobing)
        obs_grid.set_destination_grid(grid)
        obs_on_grid = obs_grid.convert(obs_seq['obs'], is_vector=self.variables[obs_name]['is_vector'], coarse_grain=True)
        ##overwrite the obs info with superobs
        if self.variables[obs_name]['is_vector']:
            msk = np.isnan(obs_on_grid[0,...])
            obs_seq['obs'] = np.array([obs_on_grid[0,~msk].flatten(), obs_on_grid[1,~msk].flatten()])
        else:
            msk = np.isnan(obs_on_grid)
            obs_seq['obs'] = obs_on_grid[~msk].flatten()
        obs_seq['x'] = grid.x[~msk].flatten()
        obs_seq['y'] = grid.y[~msk].flatten()
        ##other parameters
        for key in ('z', 't', 'err_std'):
            obs_seq[key] = np.full(obs_seq['x'].size, obs_seq[key][0])

        return obs_seq

    def random_network(self):
        raise NotImplementedError

    def get_model_files(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        ##start and end time of the trajectories
        t0 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_min']
        t1 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_max']

        if kwargs['member'] is not None:
            mstr = '{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        file_list = []
        t_list = []
        search = os.path.join(kwargs['path'], '..', '..', '*', kwargs['model_src'], mstr, 'mesh*.bin')
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

        file_list = [file_list[n] for n in np.argsort(t_list)]
        assert len(file_list)>0, 'no matching files found'

        return file_list

    def get_traj_pairs(self, **kwargs):
        """get nextsim simulated trajectory pairs, corresponding to rgps"""
        model = kwargs['model']

        ##rgps start positions on rgps_proj, in km units
        x0, y0 = kwargs['x0'], kwargs['y0']

        ##make a copy of start position to be updated to final position
        nrec = len(x0)
        x, y, i, dx, dy = [], [], [], [], []
        for r in range(nrec):
            x.append(x0[r].copy()*1000)
            y.append(y0[r].copy()*1000)
            i.append(np.zeros(x0[r].shape, dtype='int'))
            dx.append(np.zeros(x0[r].shape))
            dy.append(np.zeros(x0[r].shape))

        ##time for the start/end position
        t0 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_min']
        t1 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_max']

        file_list = self.get_model_files(**kwargs)

        for n in range(len(file_list)):
            fname = file_list[n]

            ##get model mesh points in native proj, in meters
            model.read_grid(meshfile=fname, **kwargs)
            mx = model.grid.x
            my = model.grid.y
            ##convert to rgps_proj, in km units
            x_, y_ = self.proj(*model.grid.proj(mx, my, inverse=True))

            if n==0 or 'post_regrid' in fname:
                ##build kdtree for spatial search if model mesh changed
                tree = KDTree(np.vstack([x_, y_]).T)

                ##search for traj position in mesh
                for r in range(nrec):
                    d, i[r] = tree.query(np.vstack([x[r], y[r]]).T)
                    dx[r], dy[r] = x[r] - x_[i[r]], y[r] - y_[i[r]]

            if 'post_regrid' not in fname:
                ##move to next position
                for r in range(nrec):
                    x[r], y[r] = x_[i[r]] + dx[r], y_[i[r]] + dy[r]

        pairs = []
        for r in range(nrec):
            pairs.append([x0[r], y0[r], np.full(x0[r].shape, t0),
                          x[r]/1000,  y[r]/1000, np.full(x0[r].shape, t1)])
        return pairs

    def _get_seaice_property(self, obs_name, compute_func, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
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
            obs_value = compute_func(x0, y0, t0, x1, y1, t1)

            if self.variables[obs_name]['is_vector']:
                obs_seq[:, rind] = obs_value[:, ind]
            else:
                obs_seq[rind] = obs_value[ind]
            rec += 1
        return obs_seq

    def get_seaice_drift(self, **kwargs):
        return self._get_seaice_property('seaice_drift', get_velocity, **kwargs)

    def get_seaice_deform_shear(self, **kwargs):
        return self._get_seaice_property('seaice_deform_shear', get_deform_shear, **kwargs)

    def get_seaice_deform_div(self, **kwargs):
        return self._get_seaice_property('seaice_deform_div', get_deform_div, **kwargs)

    def get_seaice_deform_vort(self, **kwargs):
        return self._get_seaice_property('seaice_deform_vort', get_deform_vort, **kwargs)
