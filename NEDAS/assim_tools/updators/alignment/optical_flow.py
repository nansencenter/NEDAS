import numpy as np
import cv2

class OpticalFlow:
    def __init__(self, method='DIS', **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, grid, fld1, fld2):
        if self.method == 'DIS':
            dis_creator = getattr(cv2, 'DISOpticalFlow_create')
            dis = dis_creator(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            frame1 = ((fld1 - np.nanmin(fld1)) / (np.nanmax(fld1) - np.nanmin(fld1)) * 255).astype(np.uint8)
            frame2 = ((fld2 - np.nanmin(fld2)) / (np.nanmax(fld2) - np.nanmin(fld2)) * 255).astype(np.uint8)
            flow = dis.calc(frame1, frame2, None)
            u, v = flow[...,0], flow[...,1]
            u *= grid.dx
            v *= grid.dy
            return np.array([u, v])

        elif self.method == 'Farneback':
            frame1 = ((fld1 - np.nanmin(fld1)) / (np.nanmax(fld1) - np.nanmin(fld1)) * 255).astype(np.uint8)
            frame2 = ((fld2 - np.nanmin(fld2)) / (np.nanmax(fld2) - np.nanmin(fld2)) * 255).astype(np.uint8)
            farneback = getattr(cv2, 'calcOpticalFlowFarneback')
            flow = farneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            u, v = flow[...,0], flow[...,1]
            u *= grid.dx
            v *= grid.dy
            return np.array([u, v])

        elif self.method == 'HornSchunk_pyramid':
            return optical_flow_HS_pyramid(grid, fld1, fld2, **self.kwargs)

        else:
            raise ValueError(f"Unsupported optical flow method: {self.method}")

def optical_flow_HS_pyramid(grid, fld1, fld2, nlevel=5, niter_max=100, smoothness_weight=1, **kwargs):
    ni = int(2**np.ceil(np.log(np.max(fld1.shape))/np.log(2)))
    x1 = np.full((ni,ni), np.nan)
    x2 = np.full((ni,ni), np.nan)
    x1[0:grid.ny, 0:grid.nx] = fld1.copy()
    x2[0:grid.ny, 0:grid.nx] = fld2.copy()
    mask = np.logical_or(np.isnan(x1), (np.abs(x2-x1)<0.00001))
    x1[mask] = 0
    x2[mask] = 0
    w = smoothness_weight
    ni, nj = x1.shape
    # normalize field so that w can be fixed
    xmax = np.max(x1[:, :]); xmin = np.min(x1[:, :])
    if (xmax>xmin):
        x1[:, :] = (x1[:, :] - xmin) / (xmax -xmin)
        x2[:, :] = (x2[:, :] - xmin) / (xmax -xmin)
    u = np.zeros((ni, nj))
    v = np.zeros((ni, nj))
    # #pyramid levels
    for lev in range(nlevel, -1, -1):
        x1w = warp(x1, -u, -v)
        x1c = coarsen(x1w, 1, lev)
        x2c = coarsen(x2, 1, lev)
        maskc = coarsen_mask(mask, 1, lev)
        xdx = 0.5*(deriv_x(x1c) + deriv_x(x2c))
        xdy = 0.5*(deriv_y(x1c) + deriv_y(x2c))
        xdt = x2c - x1c
        # #compute incremental flow using iterative solver
        du = np.zeros(xdx.shape)
        dv = np.zeros(xdx.shape)
        du1 = np.zeros(xdx.shape)
        dv1 = np.zeros(xdx.shape)
        niter = 0
        diff = 1e7
        while diff > 1e-3 and niter < niter_max:
            du[0,:] = 0; du[-1,:] = 0; du[:,0] = 0; du[:,-1] = 0  # boundary conditions
            dv[0,:] = 0; dv[-1,:] = 0; dv[:,0] = 0; dv[:,-1] = 0
            du[maskc] = 0
            dv[maskc] = 0
            ubar = laplacian(du) + du
            vbar = laplacian(dv) + dv
            du1 = ubar - xdx*(xdx*ubar + xdy*vbar + xdt)/(w + xdx**2 + xdy**2)
            dv1 = vbar - xdy*(xdx*ubar + xdy*vbar + xdt)/(w + xdx**2 + xdy**2)
            diff = np.max(np.abs(du1-du) + np.abs(dv1-dv))
            du = du1
            dv = dv1
            niter += 1
        #print(niter, diff)
        u += sharpen(du*2**(lev-1), lev, 1)
        v += sharpen(dv*2**(lev-1), lev, 1)

    disp_u = u[0:grid.ny, 0:grid.nx] * grid.dx
    disp_v = v[0:grid.ny, 0:grid.nx] * grid.dy
    return np.array([disp_u, disp_v])

def coarsen_mask(x, lev1, lev2):  # only subsample no smoothing, avoid mask growing
    if lev1 < lev2:
        for _ in range(lev1, lev2):
            ni, nj = x.shape
            x1 = x[0:ni:2, 0:nj:2]
            x = x1
    return x

def coarsen(x, lev1, lev2):
    if lev1 < lev2:
        for _ in range(lev1, lev2):
            ni, nj = x.shape
            x1 = 0.25*(x[0:ni:2, 0:nj:2] + x[1:ni:2, 0:nj:2] + x[0:ni:2, 1:nj:2] + x[1:ni:2, 1:nj:2])
            x = x1
    return x

def sharpen(x, lev1, lev2):
    if lev1 > lev2:
        for _ in range(lev1, lev2, -1):
            ni, nj = x.shape
            x1 = np.zeros((ni*2, nj))
            x1[0:ni*2:2, :] = x
            x1[1:ni*2:2, :] = 0.5*(np.roll(x, -1, axis=0) + x)
            x2 = np.zeros((ni*2, nj*2))
            x2[:, 0:nj*2:2] = x1
            x2[:, 1:nj*2:2] = 0.5*(np.roll(x1, -1, axis=1) + x1)
            x = x2
    return x

def deriv_y(f):
    return 0.5*(np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))

def deriv_x(f):
    return 0.5*(np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))

def laplacian(f):
    out = (np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) + np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0))/6
    out += (np.roll(np.roll(f, -1, axis=0), -1, axis=1) + np.roll(np.roll(f, -1, axis=0), 1, axis=1) + np.roll(np.roll(f, 1, axis=0), -1, axis=1) + np.roll(np.roll(f, 1, axis=0), 1, axis=1))/12
    out -= f
    return out

def warp(x, u, v):
    xw = x.copy()
    ni, nj = x.shape
    ii, jj = np.mgrid[0:ni, 0:nj]
    xw = interp2d(x, ii+v, jj+u)
    return xw

def interp2d(x, io, jo):
    ni, nj = x.shape
    io1 = np.floor(io).astype(int) % ni
    jo1 = np.floor(jo).astype(int) % nj
    io2 = np.floor(io+1).astype(int) % ni
    jo2 = np.floor(jo+1).astype(int) % nj
    di = io - np.floor(io)
    dj = jo - np.floor(jo)
    xo = (1-di)*(1-dj)*x[io1, jo1] + di*(1-dj)*x[io2, jo1] + (1-di)*dj*x[io1, jo2] + di*dj*x[io2, jo2]
    return xo
