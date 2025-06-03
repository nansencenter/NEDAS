import numpy as np
from matplotlib.tri import Triangulation

def get_deformation_nodes(x, y, u, v):
    tri = Triangulation(x, y)
    e1, e2, e3, tri_a, tri_p = get_deformation_on_triangulation(x, y, u, v, tri.triangles)
    return e1, e2, e3, tri_a, tri_p, tri.triangles

def get_deformation_on_triangulation(x, y, u, v, t):
    xt, yt, ut, vt = [i[t].T for i in (x, y, u, v)]
    tri_x = np.diff(np.vstack([xt, xt[0]]), axis=0)
    tri_y = np.diff(np.vstack([yt, yt[0]]), axis=0)
    tri_s = np.hypot(tri_x, tri_y)
    tri_p = np.sum(tri_s, axis=0)
    s = tri_p/2
    tri_a = np.sqrt(s * (s - tri_s[0]) * (s - tri_s[1]) * (s - tri_s[2]))
    e1, e2, e3 = get_deformation_elems(xt, yt, ut, vt, tri_a)
    return e1, e2, e3, tri_a, tri_p

def get_deformation_elems(x, y, u, v, a):
    ux = uy = vx = vy = 0
    for i0, i1 in zip([1, 2, 0], [0, 1, 2]):
        ux += (u[i0] +u[i1]) * (y[i0] - y[i1])
        uy -= (u[i0] +u[i1]) * (x[i0] - x[i1])
        vx += (v[i0] +v[i1]) * (y[i0] - y[i1])
        vy += (v[i0] +v[i1]) * (x[i0] - x[i1])
    ux, uy, vx, vy = [i / (2 * a) for i in (ux, uy, vx, vy)]
    e1 = ux + vy
    e2 = np.hypot(ux - vy, uy + vx)
    e3 = vx - uy
    return e1, e2, e3
