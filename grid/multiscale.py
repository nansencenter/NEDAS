
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

