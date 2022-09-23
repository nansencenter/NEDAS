import numpy as np

def get_grid_from_msh(msh_file):
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

def gen_uniform_grid(xstart, xend, ystart, yend, dx, rot_ang):
    RE = 6378273.
    th = rot_ang *np.pi/180.
    xcoord = np.arange(xstart, xend, dx*1e3)
    ycoord = np.arange(ystart, yend, dx*1e3)
    xi, yi = np.meshgrid(xcoord, ycoord)
    x = np.cos(th)*xi + np.sin(th)*yi
    y = -np.sin(th)*xi + np.cos(th)*yi
    z = np.array(np.sqrt(RE**2 - x**2 - y**2))
    return x, y, z

def lonlat_to_xyz(lon, lat):
    d2r = np.pi/180.
    x = np.cos(lat*d2r)*np.cos(lon*d2r)
    y = np.cos(lat*d2r)*np.sin(lon*d2r)
    z = np.sin(lat*d2r)
    return x, y, z

def xyz_to_lonlat(x, y, z):
    r2d = 180./np.pi
    radius = np.array(np.sqrt(pow(x, 2.)+pow(y, 2.)+pow(z, 2.)), dtype=float)
    lat = np.arcsin(z/radius) * r2d
    lon = np.arctan2(y, x) * r2d
    lon -= 360. * np.floor(lon / 360.)
    return lon, lat

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

