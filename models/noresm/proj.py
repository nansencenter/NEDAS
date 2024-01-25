import numpy as np
from numba import njit

@njit
def lonlat2sxy(lon, lat):
    """convert lon,lat to stereographic plane x,y"""
    invrad = np.pi / 180.
    r = np.cos((lat+90)/2*invrad)
    x = r * np.cos(lon*invrad)
    y = r * np.sin(lon*invrad)
    return x, y


@njit
def sxy2lonlat(sx, sy):
    """convert stereographic plane x,y to lon,lat"""
    rad = 180. / np.pi
    r = np.hypot(sx, sy)
    lat = 2 * rad * np.arccos(r) - 90
    lon = rad * np.arctan2(sy, sx)
    return lon, lat


@njit
def spherdist(lon1, lat1, lon2, lat2):
    """
    Calculate distance between (lon1,lat1) and (lon2,lat2)

    This function treats the Earth as a perfect sphere, it is
    not as accurate as pyproj.Geod.inv which takes the ellipse
    into account. But for the purpose of searching for positions
    in the domain, this function is accurate enough and faster.

    Inputs: lon1, lat1, lon2, lat2: float, degrees
    Output: dist: float, meters
    """
    invrad = np.pi / 180.
    Re = 6.371e6

    dlon = lon2 - lon1

    cosarc =  (np.sin(lat1*invrad) * np.sin(lat2*invrad) +
               np.cos(lat1*invrad) * np.cos(lat2*invrad) * np.cos(dlon*invrad))

    dist = Re * np.arccos(np.minimum(1., np.maximum(-1., cosarc)))

    return dist


@njit
def tri_area(a, b, c):
    """compute triangle area given edge lengths"""
    s = 0.5 * (a + b + c)
    d = s * (s-a) * (s-b) * (s-c)
    if d < 0:
        return 0  ##the triangle is not well defined
    else:
        return np.sqrt(d)


@njit
def pivotp(glon, glat, neighbors, lon, lat, ipiv=0, jpiv=0):
    """in model glon,glat, find the pivot point for lon,lat"""

    min_d = spherdist(glon[jpiv, ipiv], glat[jpiv, ipiv], lon, lat)
    while min_d > 0:
        i_c, j_c = ipiv, jpiv

        ##search in the 4 neighbors
        for n in range(4):
            j_n, i_n = neighbors[:, n, jpiv, ipiv]
            d = spherdist(glon[j_n, i_n], glat[j_n, i_n], lon, lat)
            if d < min_d:
                ipiv, jpiv = i_n, j_n
                min_d = d

        ##stop if no progress are made
        if ipiv==i_c and jpiv==j_c:
            break

    return ipiv, jpiv


@njit
def lonlat2xy(glon, glat, gx, gy, neighbors, lon, lat, ipiv=0, jpiv=0):
    """ convert from lon,lat to grid space x,y """
    ipiv, jpiv = pivotp(glon, glat, neighbors, lon, lat, ipiv, jpiv)

    pt = jpiv, ipiv
    pt_e = neighbors[0, 0, *pt], neighbors[1, 0, *pt]
    pt_n = neighbors[0, 1, *pt], neighbors[1, 1, *pt]
    pt_w = neighbors[0, 2, *pt], neighbors[1, 2, *pt]
    pt_s = neighbors[0, 3, *pt], neighbors[1, 3, *pt]
    pt_ne = neighbors[0, 1, *pt_e], neighbors[1, 1, *pt_e]
    pt_nw = neighbors[0, 1, *pt_w], neighbors[1, 1, *pt_w]
    pt_se = neighbors[0, 3, *pt_e], neighbors[1, 3, *pt_e]
    pt_sw = neighbors[0, 3, *pt_w], neighbors[1, 3, *pt_w]

    ##find which triangle (lon,lat) falls in
    ##  pt_nw --- pt_n ----- pt_ne
    ##     | .  3  .   2   .  \
    ##     | 4  .   .  .   1   \
    ##   pt_w...... pt......... pt_e
    ##     | 5   .   .   .   8   \
    ##     |  .   6   .  7    .   \
    ##  pt_sw ----- pt_s -------- pt_se

    pt_lst   = [[pt_e, pt_ne], [pt_ne, pt_n], [pt_n, pt_nw], [pt_nw, pt_w], [pt_w, pt_sw], [pt_sw, pt_s], [pt_s, pt_se], [pt_se, pt_e]]
    xoff_lst = [[   1,     1], [    1,    0], [   0,    -1], [   -1,   -1], [  -1,    -1], [   -1,    0], [   0,     1], [    1,    1]]
    yoff_lst = [[   0,     1], [    1,    1], [   1,     1], [    1,    0], [   0,    -1], [   -1,   -1], [  -1,    -1], [   -1,    0]]

    for p in range(len(pt_lst)):
        pt1 = pt
        pt2 = pt_lst[p][0]
        pt3 = pt_lst[p][1]
        xoff2 = xoff_lst[p][0]
        xoff3 = xoff_lst[p][1]
        yoff2 = yoff_lst[p][0]
        yoff3 = yoff_lst[p][1]

        d1 = spherdist(lon, lat, glon[pt], glat[pt])
        d2 = spherdist(lon, lat, glon[pt2], glat[pt2])
        d3 = spherdist(lon, lat, glon[pt3], glat[pt3])
        s1 = spherdist(glon[pt2], glat[pt2], glon[pt3], glat[pt3])
        s2 = spherdist(glon[pt3], glat[pt3], glon[pt], glat[pt])
        s3 = spherdist(glon[pt], glat[pt], glon[pt2], glat[pt2])
        A1 = tri_area(s1, d2, d3)
        A2 = tri_area(s2, d1, d3)
        A3 = tri_area(s3, d1, d2)
        Atot = tri_area(s1, s2, s3)
        if Atot == 0:  ##the triangle is not well defined
            continue

        ##find barycentric coordinates
        b1 = A1 / Atot
        b2 = A2 / Atot
        b3 = A3 / Atot
        sum_b = b1 + b2 + b3
        if np.abs(sum_b - 1) > 1e-3:   ##point falls outside this triangle
            continue

        ##the output is weighted average of x,y of triangle vertices
        x1, y1 = pt[1], pt[0]
        x2, y2 = x1+xoff2, y1+yoff2
        x3, y3 = x1+xoff3, y1+yoff3
        x = x1*b1 + x2*b2 + x3*b3
        y = y1*b1 + y2*b2 + y3*b3
        # print(ipiv, jpiv, ':', pt2, pt3, ':', b1, b2, b3, ':', x, y)

        return x, y, ipiv, jpiv

    return np.nan, np.nan, ipiv, jpiv


@njit
def xy2lonlat(glon, glat, gx, gy, neighbors, x, y):
    """ convert from grid space x,y to lon,lat """
    ##get 1d coordinates in x,y directions
    ny, nx = gx.shape
    x_, y_ = np.arange(nx+1), np.arange(ny+1)

    ##search in gx,gy for x,y
    i = np.searchsorted(x_, x, side='right')
    j = np.searchsorted(y_, y, side='right')
    inside = ~np.logical_or(np.logical_or(i==len(x_), i==0),
                            np.logical_or(j==len(y_), j==0))

    if inside:
        ##get the four vertices
        j0,i0 = j-1,i-1               ##current pivot point
        j1,i1 = neighbors[:,0,j0,i0]  ##point to the east
        j2,i2 = neighbors[:,1,j1,i1]  ##point to the north-east
        j3,i3 = neighbors[:,1,j0,i0]  ##point to the north

        ##internal coordinates for interpolation weights
        u, v = x - x_[i0], y - y_[j0]
        w = np.array([(1-u)*(1-v), u*(1-v), (1-u)*v, u*v])

        ##lon,lat at vertices, convert to stere proj, interp, convert back
        lon_pt = np.array([glon[j0,i0], glon[j1,i1], glon[j2,i2], glon[j3,i3]])
        lat_pt = np.array([glat[j0,i0], glat[j1,i1], glat[j2,i2], glat[j3,i3]])
        sx_pt, sy_pt = lonlat2sxy(lon_pt, lat_pt)
        sx, sy = np.sum(w*sx_pt), np.sum(w*sy_pt)
        lon, lat = sxy2lonlat(sx, sy)

        return lon, lat

    else:
        return np.nan, np.nan


