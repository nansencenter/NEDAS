import numpy as np
import importlib
from scipy.io import loadmat
import glob
from datetime import datetime, timedelta
import pyproj
from matplotlib.tri import Triangulation
from scipy.spatial import KDTree
from conversion import t2s

##variable dictionary for RGPS naming convention
variables = {'seaice_drift': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'km/day'},
             'seaice_shear': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
             'seaice_div': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
             'seaice_vort': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
             }

##RGPS trajectory data (x,y) is in NorthPolarStereo projection:
rgps_proj = pyproj.Proj('+proj=stere +a=6378273 +b=6356889.448910593 +lat_0=90 +lon_0=-45 +lat_ts=70')

##tolerance when searching for the time along trajectory
dt_tol=timedelta(days=2)

##some parameters
DRIFT_MAX = 80  ##km/day
DRIFT_ERR_STD = 5
DEFORM_MAX = 1  ##1/day
DEFORM_ERR_STD = 0.02

def filename(path, **kwargs):
    t = kwargs['time']

    if 'obs_window_min' in kwargs and 'obs_window_max' in kwargs:
        d_range = [kwargs['obs_window_min'], kwargs['obs_window_max']]
    else:
        d_range = [0]

    file_list = []
    for result in glob.glob(path+'/RGPS*traj.mat'):
        ss = result.split('_')
        t1 = datetime.strptime(ss[1], '%Y-%m-%d')
        t2 = datetime.strptime(ss[2], '%Y-%m-%d')
        for d in d_range:
            t_ = t + d * timedelta(hours=1)
            if t_ >= t1 and t_ <= t2 and result not in file_list:
                file_list.append(result)

    assert len(file_list)>0, 'no matching files found'

    return file_list


def read_obs(path, grid, mask, model_z, **kwargs):
    """read obs from rgps dataset"""
    obs_name = kwargs['name']
    assert obs_name in variables, 'rgps.read_obs: obs variable '+obs_name+' not defined in rgps.variables'

    ##note: x,y are obs location on grid.proj (in meters)
    ##      x0,y0,x1,y1 are position on rgps_proj (in kilometers),
    ##      triangles: velocity is defined on nodes and deform on elements
    ##      record: trajectory id, we process one record at a time
    obs_seq = {'obs':[], 'err_std':[], 't':[], 'z':[], 'y':[], 'x':[],
               'x0':[], 'y0':[], 'triangles':[], 'record':[], 'index':[]}

    rec = 0
    for file_name in filename(path, **kwargs):
        pairs = get_rgps_traj_pairs(file_name, kwargs['time'], kwargs['obs_window_min'], kwargs['obs_window_max'])

        for x0, y0, t0, x1, y1, t1 in pairs:
            tri = get_triangulation(x0, y0)
            u, v = get_velocity(x0, y0, t0, x1, y1, t1)

            ##convert x0,y0 from rgps_proj to grid.proj in meters
            x, y = grid.proj(*rgps_proj(x0*1000, y0*1000, inverse=True))

            if obs_name == 'seaice_drift':
                for p in range(len(x)):
                    ##quality check
                    if np.hypot(u[p], v[p]) > DRIFT_MAX or np.isnan(u[p]) or np.isnan(v[p]):
                        continue

                    ##add the obs to sequence
                    obs_seq['obs'].append([u[p], v[p]])
                    obs_seq['err_std'].append(DRIFT_ERR_STD)
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
                ux, uy, vx, vy = get_velocity_gradients(x0, y0, u, v)

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
                    if np.abs(obs_values[p]) > DEFORM_MAX or np.isnan(obs_values[p]):
                        continue

                    ##add the obs to sequence
                    obs_seq['obs'].append(obs_values[p])
                    obs_seq['err_std'].append(DEFORM_ERR_STD)
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

    if variables[obs_name]['is_vector']:
        obs_seq['obs'] = obs_seq['obs'].T

    return obs_seq


##utility funcs for rgps dataset
def get_rgps_traj_pairs(file_name, time, obs_window_min, obs_window_max):
    """ Get rgps trajectory pairs x,y,t defined on rgps_proj, in km,day units"""

    ##several records, each contain pairs of points:
    ##x0,y0,t0 at time and x1,y1,t1 at time+dt
    d0_out = time + timedelta(hours=1) * obs_window_min
    d1_out = time + timedelta(hours=1) * obs_window_max

    pairs = []
    for stream in loadmat(file_name)['out'][0]:

        traj_x = [i[0] for i in stream['trajectories'][0][0]['x_map'][0]]
        traj_y = [i[0] for i in stream['trajectories'][0][0]['y_map'][0]]
        year = [i[0] for i in stream['trajectories'][0][0]['year'][0]]
        day = [i[0] for i in stream['trajectories'][0][0]['time'][0]]
        traj_d = [np.array([datetime(yr,1,1)+timedelta(d) for yr,d in zip(yr_,d_)])
                  for yr_,d_ in zip(year, day)]

        xyd = ([],[],[],[],[],[])
        for x, y, d in zip(traj_x, traj_y, traj_d):
            if d[0] > d1_out or d[-1] < d0_out:
                continue  ##the target time is not in this traj
            dt0 = np.abs(d0_out - d)
            i0 = np.argmin(dt0)
            if dt0[i0] > dt_tol:
                continue
            dt1 = np.abs(d1_out - d)
            i1 = np.argmin(dt1)
            if dt1[i1] > dt_tol:
                continue

            _ = [i.append(j) for i,j in zip(xyd, [x[i0], y[i0], d[i0], x[i1], y[i1], d[i1]])]

        xyd = [np.array(i) for i in xyd]

        ##round time to days and select unique points in the pair
        rd = np.vstack([[np.round((d - d0_out).total_seconds()/3600) for d in dd] for dd in [xyd[2], xyd[5]]])
        rdu = np.unique(rd, axis=1)

        for rdu0, rdu1 in rdu.T:
            ind = np.logical_and(rd[0]==rdu0, rd[1]==rdu1)
            if len(xyd[0][ind]) < 3:  ##discard if too few points to form a mesh
                continue
            pairs.append([i[ind] for i in xyd])

    return pairs


def get_triangulation(x, y):
    """
    Get triangle indices, area, perimeter and mask
    for the RGPS trajectory points, x,y coordinates in km
    """

    tri = Triangulation(x, y)

    ##vertex coordinates
    xt, yt = [i[tri.triangles].T for i in (x, y)]

    ##side lengths
    tri_x = np.diff(np.vstack([xt, xt[0]]), axis=0)
    tri_y = np.diff(np.vstack([yt, yt[0]]), axis=0)
    tri_s = np.hypot(tri_x, tri_y)

    ##perimeter
    tri.p = np.sum(tri_s, axis=0)

    ##area
    s = tri.p / 2
    tri.a = np.sqrt(s * (s-tri_s[0]) * (s-tri_s[1]) * (s-tri_s[2]))

    ##mask off some triangles that don't belong to the mesh
    tri.mask = np.logical_or(tri.a>300, tri.p>50, tri.a/tri.p<1.5)

    return tri


def get_velocity(x0, y0, t0, x1, y1, t1):
    """get drift velocity in km/day"""
    d = ((t1 - t0) / timedelta(days=1)).astype(float)
    u, v = np.full(d.shape, np.nan), np.full(d.shape, np.nan)
    ind = (d>0)
    u[ind] = (x1[ind] - x0[ind]) / d[ind]
    v[ind] = (y1[ind] - y0[ind]) / d[ind]
    return u, v


def get_velocity_gradients(x, y, u, v):
    tri = get_triangulation(x, y)

    xt, yt, ut, vt = [i[tri.triangles].T for i in (x, y, u, v)]

    ux, uy, vx, vy = 0, 0, 0, 0
    for i0, i1 in zip([1, 2, 0], [0, 1, 2]):
        ux += (ut[i0] + ut[i1]) * (yt[i0] - yt[i1])
        uy -= (ut[i0] + ut[i1]) * (xt[i0] - xt[i1])
        vx += (vt[i0] + vt[i1]) * (yt[i0] - yt[i1])
        vy -= (vt[i0] + vt[i1]) * (xt[i0] - xt[i1])

    ux, uy, vx, vy = [i / (2 * tri.a) for i in (ux, uy, vx, vy)]

    return ux, uy, vx, vy


def random_network(path, grid, mask, z, truth_path, **kwargs):
    pass


##utility funcs for state_to_obs from nextsim model
nextsim = importlib.import_module('models.nextsim.v1')

def get_nextsim_files(path, **kwargs):
    ##start and end time of the trajectories
    t0 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_min']
    t1 = kwargs['time'] + timedelta(hours=1) * kwargs['obs_window_max']

    if 'member' in kwargs and kwargs['member'] is not None:
        mstr = '{:03d}'.format(kwargs['member']+1)
    else:
        mstr = ''

    file_list = []
    t_list = []
    search = path+'/'+mstr+'/mesh*.bin'
    for result in glob.glob(search):
        if 'final' in result:
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


def get_nextsim_traj_pairs(path, **kwargs):
    """get nextsim simulated trajectory pairs, corresponding to rgps"""

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

    file_list = get_nextsim_files(path, **kwargs)

    for n in range(len(file_list)):
        fname = file_list[n]

        ##get model mesh points in native proj, in meters
        mx = nextsim.read_data(fname, 'Nodes_x')
        my = nextsim.read_data(fname, 'Nodes_y')
        ##convert to rgps_proj, in km units
        x_, y_ = rgps_proj(*nextsim.proj(mx, my, inverse=True))
        mx, my = x_/1000, y_/1000
        ##triangle indices
        elements = nextsim.read_data(fname, 'Elements')
        mi = elements.reshape((-1, 3)) - 1

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


def get_nextsim_seaice_property(obs_name):
    def func(path, grid, mask, z, **kwargs):
        obs_name = kwargs['name']
        assert obs_name in variables, 'rgps.get_nextsim_seaice_property: unknown obs variable '+obs_name

        pairs = get_nextsim_traj_pairs(path, **kwargs)

        xo, yo, tri, r, i = kwargs['x'], kwargs['y'], kwargs['triangles'], kwargs['record'], kwargs['index']
        nobs = xo.size

        if variables[obs_name]['is_vector']:
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
    return func


##assemble the obs operators
obs_operator = {}
obs_operator['nextsim.v1'] = {'seaice_drift':get_nextsim_seaice_property('seaice_drift'),
                              'seaice_shear':get_nextsim_seaice_property('seaice_shear'),
                              'seaice_div':get_nextsim_seaice_property('seaice_div'),
                              'seaice_vort':get_nextsim_seaice_property('seaice_vort'), }


