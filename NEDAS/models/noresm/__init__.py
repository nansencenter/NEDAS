import numpy as np
import glob, os
from pyproj import Proj
from scipy.interpolate import LinearNDInterpolator
from netCDF4 import Dataset
from datetime import datetime, timedelta
from NEDAS.grid import Grid
from .proj import lonlat2xy, xy2lonlat
from NEDAS.models import Model

class NorESM(Model):
    pass

levels = np.arange(1, 31, 1)
variables = {'ocean_velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'restart_dt':1440, 'levels':levels, 'units':'m/s'},
             'ocean_temp': {'name':'temp', 'dtype':'float', 'is_vector':False, 'restart_dt':1440, 'levels':levels, 'units':'m/s'},
             'ocean_saln': {'name':'saln', 'dtype':'float', 'is_vector':False, 'restart_dt':1440, 'levels':levels, 'units':'m/s'},
            }

##staggering types for native model variables: 'p','u','v','q'
stagger = {'u':'u', 'v':'v', 'temp':'p', 'saln':'p'}

def filename(path, **kwargs):
    """
    Parse kwargs and find matching filename
    for keys in kwargs that are not set, here we define the default values
    key values in kwargs will also be checked for erroneous values
    """
    if 'time' in kwargs and kwargs['time'] is not None:
        assert isinstance(kwargs['time'], datetime), 'time shall be a datetime object'
        tstr = kwargs['time'].strftime('%Y%m%d%H%M')
    else:
        tstr = '????????'
    if 'member' in kwargs and kwargs['member'] is not None:
        assert kwargs['member'] >= 0, 'member index shall be >= 0'
        mstr = '_mem{:03d}'.format(kwargs['member']+1)
    else:
        mstr = ''

    ##get a list of filenames with matching kwargs
    search = path+'/'+'output'+tstr+mstr+'.nc'
    flist = glob.glob(search)
    assert len(flist)>0, 'no matching files found: '+search
    ##typically there will be only one matching file given the kwargs,
    ##if there is a list of matching files, we return the first one
    return flist[0]

def grid_info(grid_file, grid_type, scale_x=1, scale_y=1, stagger='p'):
    """
    Fetch grid info from a given grid_file, or generate from locally stored data.

    Args:
        grid_file (str): The path to the grid.nc file containing plat,plon...
        grid_type (str): Type of grid, 'bipolar' or 'tripolar'.
        scale_x (float): Resolution scaling in x direction.
        scale_y (float): Resolution scaling in y direction.
        stagger (str): Staggering type, 'p', 'u', 'v', or 'q'.

    Returns:
        tuple: A tuple containing:
            - lon (np.ndarray): Longitude defined on the grid points, of shape (ny, nx).
            - lat (np.ndarray): Latitude defined on the grid points
            - x (np.ndarray): X-coordinates of the grid points.
            - y (np.ndarray): Y-coordinates of the grid points.
            - neighbor (np.ndarray): Neighbor indices of shape (2, 4, ny, nx).
                For each point (:code:`j`, :code:`i`) in (ny,nx), :code:`grid_neighbors[0,:,j,i]` are the :code:`j`-indices for the
                4 neighbors (east, north, west and south) to point (:code:`j`, :code:`i`) and :code:`grid_neighbors[1,:,j,i]` are the corresponding :code:`i`-indices
    """
    ##if provided grid_file doesn't exist, try use locally stored file
    if os.path.exists(grid_file):
        filename = grid_file
    else:
        if grid_type=='bipolar':
            filename = __path__[0]+'/grid_bipolar.nc'
        elif grid_type=='tripolar':
            filename = __path__[0]+'/grid_tripolar.nc'
        else:
            raise ValueError('cannot find grid file for grid_type '+grid_type)
    # print('reading grid file: '+filename)

    with Dataset(filename) as f:
        if stagger in ['p', 'u', 'v', 'q']:
            lon = f[stagger+'lon'][...].data
            lat = f[stagger+'lat'][...].data
        else:
            raise ValueError('unknown staggering type '+stagger)

    if os.path.exists(grid_file):
        ##just get lon, lat from provided grid_file, strip padded rows
        if grid_type=='bipolar':
            grid_lon, grid_lat = lon, lat

        elif grid_type=='tripolar':
            grid_lon, grid_lat = lon[:-1, :], lat[:-1, :]

    else:
        ##refine/coarsen the lon,lat grid to the desired scaling
        ##note: this part is only for testing, usually a grid_file is provided directly
        ##      the manual refining/coarsening here is not so accurate

        ##helper proj to avoid interpolating coordinates (lon,lat) that are not continuous
        ##the north pole stereographic projection is good, since south pole
        ##is not part of the grid anyway, on hproj the coordinates are continuous
        hproj = Proj("+proj=stere +lon_0=0 +lat_0=90")

        if grid_type=='bipolar':
            ##pad orig grid to prepare for interpolation
            ny, nx = lon.shape
            lon_, lat_ = np.zeros((ny, nx+1)), np.zeros((ny, nx+1))
            lat_[:, :-1] = lat
            lat_[:, -1] = lat[:, 0]  ##cyclic east-west boundary
            lon_[:, :-1] = lon
            lon_[:, -1] = lon[:, 0]  ##cyclic east-west boundary

            hx, hy = hproj(lon_, lat_)
            xi, yi = np.meshgrid(np.arange(nx+1), np.arange(ny))
            xy2hx = LinearNDInterpolator(list(zip(xi.flatten(), yi.flatten())), hx.flatten())
            xy2hy = LinearNDInterpolator(list(zip(xi.flatten(), yi.flatten())), hy.flatten())

            x_ = np.arange(0, nx, scale_x)
            y_ = np.arange(0, ny-1+scale_y, scale_y)
            xo, yo = np.meshgrid(x_, y_)
            grid_lon, grid_lat = hproj(xy2hx(xo, yo), xy2hy(xo, yo), inverse=True)

        elif grid_type=='tripolar':
            ##pad orig grid to prepare for interpolation
            ny, nx = lon.shape
            lon_, lat_ = np.zeros((ny+1, nx+2)), np.zeros((ny+1, nx+2))
            lat_[1:, 1:-1] = lat
            lat_[1:, 0] = lat[:, -1]  ##cyclic west boundary
            lat_[1:, -1] = lat[:, 0]  ##cyclic east boundary
            lat_[0, :] = -80.2851  ##south boundary, add another latitude band
                                   ##better: extrapolated from lat[1:5,:]
            lon_[1:, 1:-1] = lon
            lon_[1:, 0] = lon[:, -1]  ##cyclic west boundary
            lon_[1:, -1] = lon[:, 0]  ##cyclic east boundary
            lon_[0, :] = lon_[1, :]   ##south boundary, add another latitude band

            hx, hy = hproj(lon_, lat_)
            xi, yi = np.meshgrid(np.arange(nx+2), np.arange(ny+1))
            xy2hx = LinearNDInterpolator(list(zip(xi.flatten(), yi.flatten())), hx.flatten())
            xy2hy = LinearNDInterpolator(list(zip(xi.flatten(), yi.flatten())), hy.flatten())

            x_ = np.arange(1+(scale_x-1)*0.5, nx+1+(scale_x-1)*0.5, scale_x)
            yc = 148   ##critical latitude 49 degree index in 0-192
            yp1 = np.arange(scale_y, yc, scale_y)
            yp2st = yp1[-1] + scale_y
            yp2ed = ny - 0.5
            yp2 = np.arange(yp2st, yp2ed, (yp2ed - scale_y/2 - yp2st)/((ny-1)/scale_y - yp1.size-1))
            y_ = np.hstack([yp1, yp2])
            xo, yo = np.meshgrid(x_, y_)
            grid_lon, grid_lat = hproj(xy2hx(xo, yo), xy2hy(xo, yo), inverse=True)

    ny, nx = grid_lon.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    grid_x, grid_y = x, y

    ##neighbor indices
    grid_neighbors = np.zeros((2, 4, ny, nx), dtype='int')
    if grid_type=='bipolar':
        ##y component
        grid_neighbors[0, 0, ...] = y                       ##east
        grid_neighbors[0, 1, :-1, :] = y[1:, :]             ##north
        grid_neighbors[0, 1, -1, :] = y[-1, :]
        grid_neighbors[0, 2, ...] = y                       ##west
        grid_neighbors[0, 3, 0, :] = y[0, :]                ##south
        grid_neighbors[0, 3, 1:, :] = y[:-1, :]
        ##x component
        grid_neighbors[1, 0, ...] = np.roll(x, -1, axis=1)  ##east
        grid_neighbors[1, 1, :, :] = x                      ##north
        grid_neighbors[1, 2, ...] = np.roll(x, 1, axis=1)   ##west
        grid_neighbors[1, 3, ...] = x                       ##south

    elif grid_type=='tripolar':
        ##y component
        grid_neighbors[0, 0, ...] = y                       ##east
        grid_neighbors[0, 1, :-1, :] = y[1:, :]             ##north
        grid_neighbors[0, 1, -1, :] = y[-2, :]
        grid_neighbors[0, 2, ...] = y                       ##west
        grid_neighbors[0, 3, 0, :] = y[0, :]                ##south
        grid_neighbors[0, 3, 1:, :] = y[:-1, :]
        ##x component
        grid_neighbors[1, 0, ...] = np.roll(x, -1, axis=1)  ##east
        grid_neighbors[1, 1, :-1, :] = x[:-1, :]            ##north
        grid_neighbors[1, 1, -1, :] = x[-1, ::-1]
        grid_neighbors[1, 2, ...] = np.roll(x, 1, axis=1)   ##west
        grid_neighbors[1, 3, ...] = x                       ##south

    return grid_lon, grid_lat, grid_x, grid_y, grid_neighbors

def read_grid(path, **kwargs):
    """ Generate a Grid object for the NorESM model grid """

    ##NorESM uses bipolar and tripolar ocean model grids
    ##See https://expearth.uib.no/?page_id=28 for some explanation
    ##See documentation: https://noresm-docs.readthedocs.io

    if 'grid_type' in kwargs:
        grid_type = kwargs['grid_type']
    else:
        grid_type = 'tripolar'  ##type of grid should match the grid.nc file

    ##scaling of the model grid, 1 correspond to the given grid_file
    ##(locally stored grids: 192x180 tripolar grid, 192x160 bipolar grid)
    ##0.5 means the resolution doubles, 2 means the resolution reduces by half
    if 'scale_x' in kwargs:
        scale_x = kwargs['scale_x']
    else:
        scale_x = 0.5  ##default
    if 'scale_y' in kwargs:
        scale_y = kwargs['scale_y']
    else:
        scale_y = 0.5  ##default

    if not 'stagger' in kwargs:
        kwargs['stagger'] = 'p'

    lon, lat, x, y, neighbors = grid_info(path+'/grid.nc', grid_type, scale_x, scale_y, kwargs['stagger'])

    ##proj function mimics a pyproj.Proj object to convert lon,lat to model x,y
    def proj(xi, yi, inverse=False):
        shape = np.atleast_1d(xi).shape
        xi, yi = np.atleast_1d(xi).flatten(), np.atleast_1d(yi).flatten()
        xo, yo = np.full(xi.size, np.nan), np.full(xi.size, np.nan)
        if not inverse:
            xo, yo = lonlat2xy(lon, lat, x, y, neighbors, xi, yi)
        else:
            xo, yo = xy2lonlat(lon, lat, x, y, neighbors, xi, yi)
        if xo.size == 1:
            return xo.item(), yo.item()
        return xo.reshape(shape), yo.reshape(shape)

    ##set some attributes for proj info
    proj.name = grid_type
    proj.definition = 'proj='+grid_type+' lon_0=70 lat_0=0'

    return Grid(proj, x, y, neighbors=neighbors)

def write_grid(path, **kwargs):
    pass

def read_var(path, grid, **kwargs):
    pass

def write_var():
    pass

