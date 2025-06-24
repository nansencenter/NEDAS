"""Utility funcs for rgps dataset"""
import numpy as np
from datetime import datetime, timedelta
from matplotlib.tri import Triangulation
from scipy.io import loadmat

def get_data_traj_pairs(file_name, d0_out, d1_out, dt_tol=2):
    """ Get rgps trajectory pairs x,y,t defined on rgps_proj, in km,day units"""

    ##several records, each contain pairs of points:
    ##x0,y0,t0 at time and x1,y1,t1 at time+dt

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

def get_deform_shear(x0, y0, t0, x1, y1, t1):
    u, v = get_velocity(x0, y0, t0, x1, y1, t1)
    ux, uy, vx, vy = get_velocity_gradients(x0, y0, u, v)
    return np.hypot(ux - vy, uy + vx)

def get_deform_div(x0, y0, t0, x1, y1, t1):
    u, v = get_velocity(x0, y0, t0, x1, y1, t1)
    ux, uy, vx, vy = get_velocity_gradients(x0, y0, u, v)
    return ux + vy

def get_deform_vort(x0, y0, t0, x1, y1, t1):
    u, v = get_velocity(x0, y0, t0, x1, y1, t1)
    ux, uy, vx, vy = get_velocity_gradients(x0, y0, u, v)
    return vx - uy
