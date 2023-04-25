import numpy as np

###DEFINITION OF REFERENCE GRID
def make_uniform_grid(xstart, xend, ystart, yend, dx):
    '''
    Make a uniform grid in Polar stereographic grid
    inputs: xstart, xend, ystart, yend: start and end points in x and y directions (in meters)
            dx: grid spacing in meters
    output: x[:, :], y[:, :] in meters
    '''
    xcoord = np.arange(xstart, xend, dx)
    ycoord = np.arange(ystart, yend, dx)
    y, x = np.meshgrid(ycoord, xcoord)
    x += 0.5*dx  ##move coords to center of grid box
    y += 0.5*dx
    return x, y

def get_theta(x, y):
    nx, ny = x.shape
    theta = np.zeros((nx, ny))
    for j in range(ny):
        dx = x[1,j] - x[0,j]
        dy = x[1,j] - y[0,j]
        theta[0,j] = np.arctan2(dy,dx)
        for i in range(1, nx-1):
            dx = x[i+1,j] - x[i-1,j]
            dy = y[i+1,j] - y[i-1,j]
            theta[i,j] = np.arctan2(dy,dx)
        dx = x[nx-1,j] - x[nx-2,j]
        dy = y[nx-1,j] - y[nx-2,j]
        theta[nx-1,j] = np.arctan2(dy,dx)
    return theta

def get_corners(x):
    nx, ny = x.shape
    xt = np.zeros((nx+1, ny+1))
    ##use linear interp in interior
    xt[1:nx, 1:ny] = 0.25*(x[1:nx, 1:ny] + x[1:nx, 0:ny-1] + x[0:nx-1, 1:ny] + x[0:nx-1, 0:ny-1])
    ##use 2nd-order polynomial extrapolat along borders
    xt[0, :] = 3*xt[1, :] - 3*xt[2, :] + xt[3, :]
    xt[nx, :] = 3*xt[nx-1, :] - 3*xt[nx-2, :] + xt[nx-3, :]
    xt[:, 0] = 3*xt[:, 1] - 3*xt[:, 2] + xt[:, 3]
    xt[:, ny] = 3*xt[:, ny-1] - 3*xt[:, ny-2] + xt[:, ny-3]
    ##make corners into new dimension
    x_corners = np.zeros((nx, ny, 4))
    x_corners[:, :, 0] = xt[0:nx, 0:ny]
    x_corners[:, :, 1] = xt[0:nx, 1:ny+1]
    x_corners[:, :, 2] = xt[1:nx+1, 1:ny+1]
    x_corners[:, :, 3] = xt[1:nx+1, 0:ny]
    return x_corners




def rotate_vector(x1, y1, u, v):
    import pynextsim.lib as nsl
    from pyproj import Proj
    dst_proj = Proj(proj='stere',
                    a=cc.RE, b=cc.RE*np.sqrt(1-cc.ECC**2),
                    lon_0=cc.LON_0, lat_0=cc.LAT_0, lat_ts=cc.LAT_TS)
    u1, v1 = nsl.transform_vectors(dst_proj, x1, y1, u, v, fill_polar_hole=True)
    return u1, v1



##nextsim
def get_unstruct_grid_from_msh(msh_file):
    '''
    Get the unstructured grid from .msh files
    output: x[:], y[:], z[:]
    '''
    f = open(msh_file, 'r')
    if "$MeshFormat" not in f.readline():
        raise ValueError("expecting $MeshFormat -  not found")
    version, fmt, size = f.readline().split()
    if "$EndMeshFormat" not in f.readline():
        raise ValueError("expecting $EndMeshFormat -  not found")
    if "$PhysicalNames" not in f.readline():
        raise ValueError("expecting $PhysicalNames -  not found")
    num_physical_names = int(f.readline())
    for _ in range(num_physical_names):
        topodim, ident, name = f.readline().split()
    if "$EndPhysicalNames" not in f.readline():
        raise ValueError("expecting $EndPhysicalNames -  not found")
    if "$Nodes" not in f.readline():
        raise ValueError("expecting $Nodes -  not found")
    num_nodes = int(f.readline())
    lines = [f.readline().strip() for n in range(num_nodes)]
    iccc =np.array([[float(v) for v in line.split()] for line in lines])
    x, y, z = (iccc[:, 1], iccc[:, 2], iccc[:, 3])
    if "$EndNodes" not in f.readline():
        raise ValueError("expecting $EndNodes -  not found")
    return x, y, z


# from .multiscale import *

##output grid to nc file
def nc_write(filename, dim, varname, dat, attr=None):
    from netCDF4 import Dataset
    '''
    write gridded data to netcdf file
    filename: path/name of the output file
    dim: dict{'dimension name': number of grid points in dimension (int)}
    varname: name for the output variable
    dat: data for output, number of dimensions must match dim
    '''
    f = Dataset(filename, 'a', format="NETCDF4_CLASSIC")
    ndim = len(dim)
    assert(len(dat.shape)==ndim)
    for key in dim:
        if key in f.dimensions:
            if dim[key]!=0:
                assert(f.dimensions[key].size==dim[key])
        else:
            f.createDimension(key, size=dim[key])
    if varname in f.variables:
        assert(f.variables[varname].shape==dat.shape)
    else:
        f.createVariable(varname, float, dim.keys())
    f[varname][:] = dat
    if attr != None:
        for akey in attr:
            f[varname].setncattr(akey, attr[akey])
    f.close()

def nc_read(filename, varname):
    from netCDF4 import Dataset
    '''
    read gridded data from netcdf file
    '''
    f = Dataset(filename, 'r')
    assert(varname in f.variables)
    dat = f[varname][:]
    f.close()
    return dat


