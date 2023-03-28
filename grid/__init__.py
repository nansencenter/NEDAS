import numpy as np
import config.constants as cc
import cartopy

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

def rotate_vector(x1, y1, u, v):
    import pynextsim.lib as nsl
    from pyproj import Proj
    dst_proj = Proj(proj='stere',
                    a=cc.RE, b=cc.RE*np.sqrt(1-cc.ECC**2),
                    lon_0=cc.LON_0, lat_0=cc.LAT_0, lat_ts=cc.LAT_TS)
    u1, v1 = nsl.transform_vectors(dst_proj, x1, y1, u, v, fill_polar_hole=True)
    return u1, v1


###MULTISCALE UTILS
##utility functions: convert between state and spectral spaces
def grid2spec(f):
    ni, nj = (f.shape[0], f.shape[1])
    return np.fft.fft2(f, axes=(0, 1))

def spec2grid(fk):
    ni, nj = (fk.shape[0], fk.shape[1])
    return np.real(np.fft.ifft2(fk, axes=(0, 1)))

##generate wavenumber sequence for fft results
def fft_wn(n):
    nup = int(np.ceil((n+1)/2))
    if n%2 == 0:
        wn = np.concatenate((np.arange(0, nup), np.arange(2-nup, 0)))
    else:
        wn = np.concatenate((np.arange(0, nup), np.arange(1-nup, 0)))
    return wn

##generate meshgrid wavenumber for input field x
## the first two dimensions are horizontal (i, j)
def get_wn(x):
    dims = x.shape
    ii = fft_wn(dims[0])
    jj = fft_wn(dims[1])
    wni = np.expand_dims(ii, 1)
    wni = np.repeat(wni, dims[1], axis=1)
    wnj = np.expand_dims(jj, 0)
    wnj = np.repeat(wnj, dims[0], axis=0)
    for d in range(2, len(dims)):  ##extra dimensions
        wni = np.expand_dims(wni, d)
        wni = np.repeat(wni, dims[d], axis=d)
        wnj = np.expand_dims(wnj, d)
        wnj = np.repeat(wnj, dims[d], axis=d)
    return wni, wnj

##scale decomposition
def lowpass_resp(Kh, k1, k2):
    r = np.zeros(Kh.shape)
    r[np.where(Kh<k1)] = 1.0
    r[np.where(Kh>k2)] = 0.0
    ind = np.where(np.logical_and(Kh>=k1, Kh<=k2))
    r[ind] = np.cos((Kh[ind] - k1)*(0.5*np.pi/(k2 - k1)))**2
    return r

def get_scale_resp(Kh, kr, s):
    ns = len(kr)
    resp = np.zeros(Kh.shape)
    if ns > 1:
        if s == 0:
            resp = lowpass_resp(Kh, kr[s], kr[s+1])
        if s == ns-1:
            resp = 1 - lowpass_resp(Kh, kr[s-1], kr[s])
        if s > 0 and s < ns-1:
            resp = lowpass_resp(Kh, kr[s], kr[s+1]) - lowpass_resp(Kh, kr[s-1], kr[s])
    return resp


def get_scale(x, kr, s):
    xk = grid2spec(x)
    xkout = xk.copy()
    ns = len(kr)
    if ns > 1:
        kx, ky = get_wn(x)
        Kh = np.sqrt(kx**2 + ky**2)
        xkout = xk * get_scale_resp(Kh, kr, s)
    return spec2grid(xkout)

##power spectra
def pwrspec2d(field):
    '''
    horizontal 2D spectrum p(k), k=sqrt(kx^2+ky^2)
    input: field[nx, ny]
    output: wn[nk], pwr[nk]
    '''
    nx, ny = field.shape
    nupx = int(np.ceil((nx+1)/2))
    nupy = int(np.ceil((ny+1)/2))
    nup = max(nupx, nupy)
    wnx = fft_wn(nx)
    wny = fft_wn(ny)
    ky, kx = np.meshgrid(wny, wnx)
    k2d = np.sqrt((kx*(nup/nupx))**2 + (ky*(nup/nupy))**2)
    FT = np.fft.fft2(field)
    P = (np.abs(FT)/nx/ny)**2
    wn = np.arange(0.0, nup)
    pwr = np.zeros((nup,))
    for w in range(nup):
        pwr[w] = np.sum(P[np.where(np.ceil(k2d)==w)])
    return wn, pwr


##make reference grid from config_file
x_ref, y_ref = make_uniform_grid(cc.XSTART, cc.XSTART+cc.NX*cc.DX,
                                 cc.YSTART, cc.YSTART+cc.NY*cc.DX,
                                 cc.DX)

crs = cartopy.crs.NorthPolarStereo(central_longitude=cc.LON_0,
                                   true_scale_latitude=cc.LAT_TS)

