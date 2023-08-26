import numpy as np
from scipy.io import loadmat
import glob
from datetime import datetime, timedelta
import pyproj

##variable dictionary for RGPS naming convention
variables = {'seaice_drift': {'dtype':'float', 'is_vector':True, 'z_type':'z', 'units':'m/s'},
             'seaice_deform_shear': {'dtype':'float', 'is_vector':False, 'z_type':'z', 'units':'1/s'},
             'seaice_deform_div': {'dtype':'float', 'is_vector':False, 'z_type':'z', 'units':'1/s'},
             }

def filename(path, **kwargs):
    flist = []
    for f in glob.glob(path+'/RGPS*traj.mat'):
        ss = f.split('_')
        t1 = datetime.strptime(ss[1], '%Y-%m-%d')
        t2 = datetime.strptime(ss[2], '%Y-%m-%d')
        if 'time' in kwargs:
            t = kwargs['time']
            if t >= t1 and t <= t2:
                flist.append(f)
        else:
            flist.append(f)
    assert len(flist)>0, 'no matching files found'
    return flist[0]

def get_traj_pairs(path, grid, **kwargs):
    f = loadmat(filename(path, **kwargs))

    ##RGPS trajectory data (x,y) is in NorthPolarStereo projection:
    rgps_proj = pyproj.Proj('+proj=stere +a=6378273 +b=6356889.448910593 +lat_0=90 +lon_0=-45 +lat_ts=70')

    streams = []
    for i in range(len(f['out'][0])):
        stream = f['out'][0][i]
        x = [i[0]*1000 for i in stream['trajectories'][0][0]['x_map'][0]]
        y = [i[0]*1000 for i in stream['trajectories'][0][0]['y_map'][0]]
        time = [i[0] for i in stream['trajectories'][0][0]['time'][0]]
        year = [i[0] for i in stream['trajectories'][0][0]['year'][0]]
        q = [i[0] for i in stream['trajectories'][0][0]['q_flag'][0]]
        d = [np.array([datetime(yi,1,1) + timedelta(ti) for (yi,ti) in zip(y_vec, t_vec)])
            for (y_vec, t_vec) in zip(year, time)]

        ##convert from rgps_proj to grid.prj
        x_new = []
        y_new = []
        for x0,y0 in zip(x,y):
            lon, lat = rgps_proj(x0, y0, inverse=True)
            x1, y1 = grid.proj(lon, lat)
            x_new.append(x1)
            y_new.append(y1)

        streams.append({'x':x_new, 'y':y_new, 'd':d, 'q':q})

    traj_pairs = []
    for stream in streams:
        for xt, yt, dt in zip(stream['x'], stream['y'], stream['d']):
            for x1, y1, d1, x2, y2, d2 in zip(xt[:-1], yt[:-1], dt[:-1], xt[1:], yt[1:], dt[1:]):
                traj_pairs.append([x1, x2, y1, y2, d1, d2])

    return traj_pairs

##read dataset and obtain a list of obs for the given kwargs (variable name, time, etc.)
##inputs: path: directory where the dataset files are stored
##        grid:
# def read_obs(path, **kwargs):
#     obs_seq = []
#     for i in range(3):
#         rec = {'x': ,
#                'y': ,
#                'z': ,
#                't': ,
#                'value': ,}
#         obs_seq.append(rec)
#     return obs_seq


# from .state_to_obs import state_to_obs

