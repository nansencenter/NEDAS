##adapted from nextsim-tools/pynextsim/lib.py

import numpy as np
from scipy.interpolate import griddata
from pyproj import Proj

def transform_vectors_weights(dst_proj, x, y, src_proj=None, inverse=False):
    '''
    get weights to rotate velocity from coordinate system defined by src_proj to that defined by dst_proj
    - if src_proj is None, it is lon-lat coords
    - if inverse=True, the velocities are relative to dst_proj instead of src_proj

    Parameters:
    -----------
    dst_proj : pyproj.Proj
        projection that maps lon,lat to x1,y1
    x : np.ndarray
        x coordinates
    y : np.ndarray
        y coordinates
    src_proj : pyproj.Proj
        projection that maps lon,lat to x2,y2. If None it is lon-lat coords
    inverse : bool
        False: u, v are relative to src_proj, want relative to dst_proj (default)
        True: u, v are relative to dst_proj, want relative to src_proj

    Returns:
    --------
    rotation_matrix : tuple(numpy.ndarray)
        (a, b, c, d) where the rotation matrix is [[a,b], [c,d]]
    gd : numpy.ndarray(bool)
    '''
    assert(isinstance(dst_proj, Proj))
    shp = x.shape
    assert(y.shape == shp)
    u2 = np.zeros(shp)+np.nan
    v2 = np.zeros(shp)+np.nan
    gd = np.ones(shp, dtype=bool)
    lon, lat = dst_proj(x, y, inverse=True)
    if src_proj is None:
        # move by "eps" degrees to determine the direction of movement in x-y plane
        eps = .05
        latmax = 89.9 # u,v nans if lat>latmax
        gd = lat<=latmax
        xu, yu = dst_proj(lon + eps, lat) #move a bit to east
        xv, yv = dst_proj(lon, lat + eps) #move a bit to north
    else:
        assert(isinstance(src_proj, Proj))
        x2, y2 = src_proj(lon, lat)
        # move by "eps" metres in x2-y2 plane to determine the direction of movement in x-y plane
        eps = 10
        lon, lat = src_proj(x2 + eps, y2, inverse=True) #move a bit in x dirn (src coords)
        xu, yu = dst_proj(lon, lat)
        lon, lat = src_proj(x2, y2 + eps, inverse=True) #move a bit in y dirn (src coords)
        xv, yv = dst_proj(lon, lat)
    def get_cos_sin(dx, dy):
        h = np.hypot(dx, dy)
        return dx/h, dy/h
    # cos, sin of direction in x,y when moving in east/x dirn
    a, c = get_cos_sin(xu[gd]-x[gd], yu[gd]-y[gd])
    # cos, sin of direction in x,y when moving in north/y dirn
    b, d = get_cos_sin(xv[gd]-x[gd], yv[gd]-y[gd])
    # if not inverse, total speed is:
    #   u*(cos_u,sin_u) + v*(cos_v,sin_v)
    #   = [[a,b],[c,d]]*[u;v]
    if not inverse:
        return (a, b, c, d), gd
    # inverse the matrix if wanting to go from dst_proj to src_proj
    det = a*d - b*c
    return (d/det, -b/det, -c/det, a/det), gd


def transform_vectors(dst_proj, x, y, u, v, src_proj=None, inverse=False, fill_polar_hole=False):
    '''
    rotate velocity from coordinate system defined by src_proj to that defined by dst_proj
    - if src_proj is None, it is lon-lat coords
    - if inverse=True, the velocities are relative to dst_proj instead of src_proj

    Parameters:
    -----------
    dst_proj : pyproj.Proj
        projection that maps lon,lat to x1,y1
    x : np.ndarray
        x coordinates
    y : np.ndarray
        y coordinates
    u : np.ndarray
        east-west or x velocity
    v : np.ndarray
        north-south or y velocity
    src_proj : pyproj.Proj
        projection that maps lon,lat to x2,y2. If None it is lon-lat coords
    inverse : bool
        False: u, v are relative to src_proj, want relative to dst_proj (default)
        True: u, v are relative to dst_proj, want relative to src_proj
    fill_polar_hole : bool
        True:
        - interpolate from lower latitudes to fill the polar hole
        - only works if interpolating from lon-lat coords to x-y coords

    Returns:
    --------
    u2 : np.ndarray
        velocity in x dirn
    v2 : np.ndarray
        velocity in y dirn
    '''
    shp = x.shape
    assert(y.shape == shp)
    assert(u.shape == shp)
    assert(v.shape == shp)
    if fill_polar_hole:
        assert(src_proj is None and not inverse)
    u2 = np.zeros(shp)+np.nan
    v2 = np.zeros(shp)+np.nan
    (a, b, c, d), gd = transform_vectors_weights(
            dst_proj, x, y, src_proj=src_proj, inverse=inverse)
    u2[gd] = a*u[gd] + b*v[gd]
    v2[gd] = c*u[gd] + d*v[gd]

    if fill_polar_hole and not np.all(gd):
        _, lat = dst_proj(x, y, inverse=True)
        gap = ~gd
        latmin, = np.percentile(lat[gd].flatten(), [90])
        gd *= lat>np.min([89, latmin]) # don't try to interpolate the whole array
        xy_gd = np.vstack([x[gd].flatten(), y[gd].flatten()]).T
        xy_gap = (x[gap].flatten(), y[gap].flatten())
        for arr in [u2, v2]:
            arr[gap] = griddata(xy_gd, arr[gd], xy_gap)
    return u2, v2
