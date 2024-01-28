import numpy as np
import importlib
from scipy.io import loadmat
import glob
from datetime import datetime, timedelta
import pyproj
from matplotlib.tri import Triangulation

##variable dictionary for RGPS naming convention
variables = {'seaice_drift': {'dtype':'float', 'is_vector':True, 'z_units':'m', 'units':'km/day'},
             'seaice_shear': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
             'seaice_div': {'dtype':'float', 'is_vector':False, 'z_units':'m', 'units':'1/day'},
             }

##RGPS trajectory data (x,y) is in NorthPolarStereo projection:
proj = pyproj.Proj('+proj=stere +a=6378273 +b=6356889.448910593 +lat_0=90 +lon_0=-45 +lat_ts=70')


def filename(path, **kwargs):
    if 'time' not in kwargs:
        t = datetime(2007, 1, 1)
    else:
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
    assert 'name' in kwargs, 'rgps.read_obs: missing obs variable name'
    assert kwargs['name'] in variables, 'rgps.read_obs: obs variable '+kwargs['name']+' not defined in rgps.variables'

    obs_seq = {'obs':[], 't':[], 'z':[], 'y':[], 'x':[], 'err_std':[],
               't1':[], 'y1':[], 'x1':[], 'record':[]}

    for file_name in filename(path, **kwargs):
        pairs = get_traj_pairs(file_name, time)

        # for pair in pairs:
        #     for x0_, y0_, t0_, x1_, y1_, t1_ in pair:

                ##convert from proj to grid.proj

                # if kwargs['name'] == 'seaice_drift':


                # elif kwargs['name'] == 'seaice_deform


                # obs_seq['obs'].append()


    return obs_seq


def random_network(path, grid, mask, z, truth_path, **kwargs):
    pass


##utility funcs
def get_rgps_traj_pairs(file_name, time):
    """get trajectory pairs from rgps dataset"""
    ##several records, each contain pairs of points:
    ##x0,y0,t0 at time and x1,y1,t1 at time+dt
    dt = timedelta(days=3)  ##RGPS records the position every 3 days
    d0_out = time
    d1_out = time+dt

    ##tolerance when searching for the time along trajectory
    dt_tol=timedelta(days=1)

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

        pair = []
        for rdu0, rdu1 in rdu.T:
            ind = np.logical_and(rd[0]==rdu0, rd[1]==rdu1)
            if len(xyd[0][ind]) < 3:  ##discard if too few points to form a mesh
                continue
            pair.append([i[ind] for i in xyd])

        pairs.append(pair)

    return pairs


def get_tri(x, y):
    """get triangle indices, area, perimeter and mask"""

    tri = Triangulation(x, y)

    ##vertex coordinates
    xt, yt = [i[tri.triangles].T for i in (x, y)]

    ##side lengths
    tri_x = np.diff(np.vstack([xt, xt[0]]), axis=0)
    tri_y = np.diff(np.vstack([yt, yt[0]]), axis=0)
    tri_s = np.hypot(tri_x, tri_y)

    ##perimeter
    tri_p = np.sum(tri_s, axis=0)

    ##area
    s = tri_p / 2
    tri_a = np.sqrt(s * (s-tri_s[0]) * (s-tri_s[1]) * (s-tri_s[2]))

    ##mask off some triangles
    ratio = tri_a / tri_p
    tri_mask = np.logical_or(tri_a>500, tri_p>100, ratio<1.5)

    return tri.triangles, tri_a, tri_p, tri_mask


def get_velocity(x0, y0, t0, x1, y1, t1):
    """get drift velocity in km/day"""
    d = ((t1 - t0) / timedelta(days=1)).astype(float)
    u = (x1 - x0) / d
    v = (y1 - y0) / d
    return u, v


def get_velocity_gradients(x, y, u, v):
    tri_i, tri_a, tri_p, mask = get_tri(x, y)

    xt, yt, ut, vt = [i[tri_i].T for i in (x, y, u, v)]

    ux, uy, vx, vy = 0, 0, 0, 0
    for i0, i1 in zip([1, 2, 0], [0, 1, 2]):
        ux += (ut[i0] + ut[i1]) * (yt[i0] - yt[i1])
        uy -= (ut[i0] + ut[i1]) * (xt[i0] - xt[i1])
        vx += (vt[i0] + vt[i1]) * (yt[i0] - yt[i1])
        vy -= (vt[i0] + vt[i1]) * (xt[i0] - xt[i1])

    ux, uy, vx, vy = [i / (2 * tri_a) for i in (ux, uy, vx, vy)]

    return ux, uy, vx, vy


obs_operator = {}

def get_nextsim_traj_pairs(path, grid, **kwargs):
    model = importlib.import_module('models.nextsim.v1')


def get_nextsim_seaice_drift(path, grid, mask, z, **kwargs):
    return obs_seq


def get_nextsim_seaice_deform_shear(path, grid, mask, z, **kwargs):
    return obs_seq


def get_nextsim_seaice_deform_div(path, grid, mask, z, **kwargs):
    return obs_seq


obs_operator['nextsim.v1'] = {'seaice_drift':get_nextsim_seaice_drift,
                              'seaice_deform_shear':get_nextsim_seaice_deform_shear,
                              'seaice_deform_div':get_nextsim_seaice_deform_div, }
