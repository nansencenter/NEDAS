import numpy as np
from scipy.interpolate import interp1d
from utils.njit import njit

# def local_factor(cov, dist, roi, localize_type):
# def local_factor(state_data, obs_data, c.localization):
#     lfactor = np.ones(cov.shape)
#     assert cov.shape == dist.shape, 'local_factor error: cov and dist shape mismatch'
#     return lfactor.reshape(shape)

@njit(cache=True)
def local_factor(dist, roi, localize_type='GC'):
# def local_factor_distance_based(dist, roi, localize_type='GC'):
    """
    Localization factor based on distance and radius of influence (roi)

    Inputs:
    - dist: np.array
      Distance between observation and state (being updated)

    - roi: float or np.array same shape as dist
      The radius of influence, distance beyond which local_factor is tapered to 0

    Return:
    - lfactor: np.array
      The localization factor, same shape as dist
    """

    shape = dist.shape
    dist = dist.flatten()
    lfactor = np.zeros(dist.shape)

    if localize_type == 'GC': ##Gaspari-Cohn function (default)
        r = dist / (roi / 2)

        ind1 = np.where(r<1)
        loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
        lfactor[ind1] = loc1[ind1]

        ind2 = np.where(np.logical_and(r>=1, r<2))
        r[np.where(r==0)] = 1e-10  ##avoid divide by 0
        loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
        lfactor[ind2] = loc2[ind2]

    elif localize_type == 'step':  #step function from 1 to 0 at roi
        lfactor[np.where(dist<=roi)] = 1.0

    elif localize_type == 'exp':  ##exponential decay
        lfactor = np.exp(-dist/roi)

    else:
        print('Error: unknown localization function type: '+localize_type)
        raise ValueError

    return lfactor.reshape(shape)

@njit(cache=True)
def local_factor_NICE(CorrXY, fac=1):
    """
    localization based on NICE (Morzfeld et al. 2023)
    """
    # Ne = X.shape[1]
    ##lookup table
    # FileName = f'std_ro_Ne_{Ne}.mat'
    # dat = scipy.io.loadmat(f'matlab/std_ro_Ne_{Ne}.mat')
    # r = dat['r'].flatten()
    # stdCrs = dat['stdCrs'].flatten()
    # CorrXY = np.corrcoef(X, Y)[0:X.shape[0], X.shape[0]:]
    interp_func = interp1d(r, stdCrs, kind='linear', fill_value='extrapolate')
    std_rho = interp_func(CorrXY)
    std_rho[np.isclose(CorrXY, 1)] = 0
    sig_rho = np.sqrt(np.sum(std_rho ** 2))
    go = True
    expo2 = 0
    while go:
        expo2 += 2
        L = np.abs(CorrXY) ** expo2
        Corr_NICER = L * CorrXY
        if np.linalg.norm(Corr_NICER - CorrXY, 'fro') > fac * sig_rho:
            go = False
    expo1 = expo2 - 2
    rho_exp1 = CorrXY ** expo1
    rho_exp2 = CorrXY ** expo2
    al = np.arange(0.1, 1.1, 0.1)
    PrevCorr = CorrXY
    for kk in range(len(al)):
        L = (1 - al[kk]) * rho_exp1 + al[kk] * rho_exp2
        Corr_NICE = L * CorrXY
        if kk > 0 and np.linalg.norm(Corr_NICER - CorrXY, 'fro') > fac * sig_rho:
            Corr_NICE = PrevCorr
            break
        elif np.linalg.norm(Corr_NICE - CorrXY, 'fro') > fac * sig_rho:
            break
        PrevCorr = Corr_NICE
    # Vy = np.diag(np.std(Y, axis=1))
    # Vx = np.diag(np.std(X, axis=1))
    # Cov_NICE = np.dot(Vx, np.dot(Corr_NICE, Vy))
    return Corr_NICE

# def NICE_lookup_table(c, nens):
    # return table

# def NICE_generate_lookup_table(filename, nens):
#     nos = 1e5;
# rTMP = 0:.05:.95;
# for Ne = 20;%[5 10 20 30 40 60 80 100]
#     fprintf('Ne = %g\n',Ne)
#     stdCrsTMP = zeros(length(rTMP),1);
#     for kk=1:length(rTMP)
#         CrTMP = zeros(nos,1);
#         for jj=1:nos
#             C = eye(2);
#             C(1,2) = rTMP(kk);
#             C(2,1) = rTMP(kk);
#             sC = chol(C)';
#             samps = sC*randn(2,Ne);
#             tmp = corr(samps');
#             CrTMP(jj) = tmp(1,2);
#         end
#         stdCrsTMP(kk) = std(CrTMP);
#     end 

#     r = [rTMP 1]; 
#     stdCrs = [stdCrsTMP; 0]; 
#     FileName = strcat('std_ro_Ne_',num2str(Ne),'.mat');
#     save(FileName,'r','stdCrs')
# end


