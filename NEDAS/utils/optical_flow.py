import numpy as np
import cv2

class OpticalFlow:
    def __init__(self, method='DIS', **kwargs):
        self.method = method
        self.kwargs = kwargs

    @staticmethod
    def _to_uint8_pair(fld1, fld2):
        """Convert a pair of fields to uint8 images for cv2's DIS/Farneback (both require 8-bit
        input -- confirmed via direct testing that float32 input raises a hard assertion error
        in cv2, so this quantization is unavoidable for these two backends specifically).

        2026-07-08 fix: previously each field was normalized independently
        (fld - fld.min())/(fld.max()-fld.min()), so the SAME physical value could map to
        DIFFERENT pixel intensities between frame1/frame2 whenever their min/max differed
        (which they generally do, being different ensemble members/times) -- this directly
        violates the brightness-constancy assumption both DIS and Farneback rely on, degrading
        their accuracy independent of any real displacement. Now uses a single shared min/max
        (the combined range of both fields) so the same physical value always maps to the same
        pixel intensity in both frames, matching Horn-Schunck's own convention of normalizing
        both frames with one shared range (there using field1's range only; combined range used
        here instead so neither frame clips/wraps if their ranges differ).
        """
        vmin = min(np.nanmin(fld1), np.nanmin(fld2))
        vmax = max(np.nanmax(fld1), np.nanmax(fld2))
        scale = 255.0 / (vmax - vmin) if vmax > vmin else 0.0
        frame1 = np.clip((fld1 - vmin) * scale, 0, 255).astype(np.uint8)
        frame2 = np.clip((fld2 - vmin) * scale, 0, 255).astype(np.uint8)
        return frame1, frame2

    def __call__(self, grid, fld1, fld2):
        if self.method == 'DIS':
            # 2026-07-08: exposed DIS's own tunable parameters (previously hardcoded to
            # PRESET_FAST with no further tuning) -- preset defaults to FAST for backward
            # compatibility, but patch_size/patch_stride/finest_scale can now be set directly.
            # Visual diagnostics (see qg_benchmark/optflow_algorithm_comparison*.png) showed
            # PRESET_FAST's default (large) patch size produces a blocky, over-smoothed
            # displacement field relative to Horn-Schunck -- smaller patch_size or
            # PRESET_MEDIUM should recover finer spatial detail, at the cost of more noise/compute.
            preset_name = self.kwargs.get('preset', 'DISOPTICAL_FLOW_PRESET_FAST')
            DISOpticalFlow_create = getattr(cv2, 'DISOpticalFlow_create')
            dis = DISOpticalFlow_create(getattr(cv2, preset_name))
            if 'finest_scale' in self.kwargs:
                dis.setFinestScale(self.kwargs['finest_scale'])
            if 'patch_size' in self.kwargs:
                dis.setPatchSize(self.kwargs['patch_size'])
            if 'patch_stride' in self.kwargs:
                dis.setPatchStride(self.kwargs['patch_stride'])
            if 'grad_descent_iter' in self.kwargs:
                dis.setGradientDescentIterations(self.kwargs['grad_descent_iter'])
            if 'variational_refine_iter' in self.kwargs:
                dis.setVariationalRefinementIterations(self.kwargs['variational_refine_iter'])
            frame1, frame2 = self._to_uint8_pair(fld1, fld2)
            flow = dis.calc(frame1, frame2, None)
            u, v = flow[...,0], flow[...,1]
            u *= grid.dx
            v *= grid.dy
            return np.array([u, v])

        elif self.method == 'Farneback':
            # 2026-07-08: exposed Farneback's own parameters (previously hardcoded). Defaults
            # match the prior hardcoded call for backward compatibility. `winsize` (averaging
            # window) is the main smoothing control -- OpenCV's own docs note larger winsize
            # "yields more blurred motion field"; the default 15 is larger than even the S-scale's
            # character_length (6.4), which likely over-smooths fine-scale displacement structure.
            calcOpticalFlowFarneback = getattr(cv2, 'calcOpticalFlowFarneback')
            frame1, frame2 = self._to_uint8_pair(fld1, fld2)
            pyr_scale = self.kwargs.get('pyr_scale', 0.5)
            levels = self.kwargs.get('levels', 3)
            winsize = self.kwargs.get('winsize', 15)
            iterations = self.kwargs.get('iterations', 3)
            poly_n = self.kwargs.get('poly_n', 5)
            poly_sigma = self.kwargs.get('poly_sigma', 1.2)
            flags = self.kwargs.get('flags', 0)
            flow = calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale, levels, winsize,
                                            iterations, poly_n, poly_sigma, flags)
            u, v = flow[...,0], flow[...,1]
            u *= grid.dx
            v *= grid.dy
            return np.array([u, v])

        elif self.method == 'HornSchunck_pyramid':
            return optical_flow_HS_pyramid(grid, fld1, fld2, **self.kwargs)

        else:
            raise ValueError(f"Unsupported optical flow method: {self.method}")

def optical_flow_HS_pyramid(grid, fld1, fld2, nlevel=5, niter_max=100, smoothness_weight=None,
                             alpha_squared=None, local_weight=None, **kwargs):
    """smoothness_weight is applied to a field ALREADY normalized to [0,1] below (using THIS
    call's own xmax-xmin, recomputed fresh every call) -- so a fixed smoothness_weight only
    matches the paper's raw-field alpha^2 for field ranges close to whatever R was used to derive
    it (2026-07-08: smoothness_weight=alpha^2/R^2, R~52.7 measured once from a cp2-scale field).
    That fixed value silently drifts wrong whenever the ACTUAL field range differs (2026-07-09:
    confirmed at cp10's longer, more nonlinear cycling period, where larger forecast divergence
    means R is genuinely larger -- the same fixed 0.036 becomes an effectively much stronger
    alpha^2, over-smoothing and collapsing MSA's performance below even plain MS). Pass
    alpha_squared (the paper's true, field-range-independent physical constant, =100) instead of
    smoothness_weight to have this function derive the correctly-scaled weight itself from the
    SAME xmax/xmin it already computes for normalization, self-consistently, every call.
    smoothness_weight is kept as a direct override for callers that want the old fixed-value
    behavior (e.g. DIS/Farneback callers never touch this function at all).

    local_weight (2026-07-20, s in [0,1)): self-normalizing alternative to alpha_squared/
    smoothness_weight, added after both were found to land in wildly different effective
    regimes across models -- alpha_squared=100 (its own field-range-independent physical
    constant) gave w/mean(data_term) ratios of 1517 (vort2d) vs 5022 (vort3d) at the coarsest
    pyramid level, because neither parameter accounts for how the actual competing quantity
    (xdx^2+xdy^2, the local squared gradient, i.e. the data term the smoothness weight w
    competes against in the solver's denominator) depends on field roughness/resolution/pyramid
    level -- only on the field's global value range. local_weight instead computes, AT EACH
    PYRAMID LEVEL, d_ref = mean(xdx^2+xdy^2) over unmasked pixels at that level, then sets
    w = d_ref * s/(1-s). Since the solver's local response scales as d(x,y)/(w+d(x,y)), at a
    TYPICAL-strength location (d=d_ref) this makes s directly mean "fraction of the typical-
    strength local response suppressed": s=0 -> no smoothing (pure local fit, noisy); s=0.5 ->
    half the typical response passes; s->1 -> full suppression (matches what alpha_squared=100/
    smoothness_weight=0.3 both did for vort3d). Stronger-than-typical local features (e.g. a
    coherent vortex core) still get a proportionally larger fraction through even at high s --
    naturally adaptive, unlike a single global w. Takes priority over alpha_squared/
    smoothness_weight if given.
    """
    ni = int(2**np.ceil(np.log(np.max(fld1.shape))/np.log(2)))
    x1 = np.full((ni,ni), np.nan)
    x2 = np.full((ni,ni), np.nan)
    x1[0:grid.ny, 0:grid.nx] = fld1.copy()
    x2[0:grid.ny, 0:grid.nx] = fld2.copy()
    mask = np.logical_or(np.isnan(x1), (np.abs(x2-x1)<0.00001))
    x1[mask] = 0
    x2[mask] = 0
    ni, nj = x1.shape
    # normalize field so that w can be fixed
    xmax = np.max(x1[:, :]); xmin = np.min(x1[:, :])
    if local_weight is not None:
        w = None  # computed fresh per pyramid level below, from that level's own data term
    elif alpha_squared is not None:
        w = alpha_squared / (xmax - xmin)**2 if xmax > xmin else alpha_squared
    else:
        w = smoothness_weight if smoothness_weight is not None else 1
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
        if local_weight is not None:
            data_term = xdx**2 + xdy**2
            active = ~maskc
            d_ref = data_term[active].mean() if np.any(active) else data_term.mean()
            s = local_weight
            w_lev = d_ref * s / (1 - s) if d_ref > 0 else 0.0
        else:
            w_lev = w
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
            du1 = ubar - xdx*(xdx*ubar + xdy*vbar + xdt)/(w_lev + xdx**2 + xdy**2)
            dv1 = vbar - xdy*(xdx*ubar + xdy*vbar + xdt)/(w_lev + xdx**2 + xdy**2)
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
