##the map projection used in nextsim
import numpy as np
from grid import Grid
import pyproj
proj = pyproj.Proj(proj='stere', a=6378273, b=6356889.448910593, lat_0=90., lon_0=-45., lat_ts=60.)

from .basic_io import get_info, read_data, write_data
from matplotlib.tri import Triangulation
from grid import Grid

def indices(filename):
    elements = read_data(filename.replace('field','mesh'), 'Elements')
    ne = int(elements.size/3)
    ind = elements.reshape((ne, 3)) - 1
    return ind

def nodes_xy(filename):
    xn = read_data(filename.replace('field','mesh'), 'Nodes_x')
    yn = read_data(filename.replace('field','mesh'), 'Nodes_y')
    return xn, yn

def elements_xy(filename):
    xn, yn = nodes_xy(filename)
    ind = indices(filename)
    xe = np.mean(xn[ind], axis=1)
    ye = np.mean(yn[ind], axis=1)
    return xe, ye

def triangulation(filename):
    xn, yn = nodes_xy(filename)
    ind = indices(filename)
    return Triangulation(xn, yn, ind)

def get_grid(filename):
    x, y = nodes_xy(filename)
    tri = triangulation(filename)
    return Grid(proj, x, y, regular=False, triangles=tri.triangles)

