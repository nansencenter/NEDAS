import numpy as np

###Define reference grid for analysis
def uniform_grid(xstart, xend, ystart, yend, dx):
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

def theta(x, y):
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

def corners(x):
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
