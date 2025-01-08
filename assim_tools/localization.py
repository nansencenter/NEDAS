"""Module for handling covariance between variables, and localization of the covariance"""
import numpy as np
from utils.njit import njit

# def parse_localization_type(ltype_str):
#     """
#     Parse the configuration string c.localization['?type']
#     which is formated as a string of localize_type(s) separated by commas
#     """
#     ltypes = ltype_str.split(',')


# def local_factor(localize_type, distance, roi, correlation):
#     """Top-level function for performing covariance localization
#     Inputs:
#     - localize_type: str
#       The type of localization to perform
#     - distance: np.array
#       The distance between the observation and the state being updated
#     - roi: float or np.array same shape as distance
#       The radius of influence, distance beyond which local_factor is tapered to 0
#     - correlation: np.array
#       The correlation between the observation and the state being updated"""
#     if is_distance_based(localize_type):
#         ind = np.where(distance < roi)
#         lfactor = local_factor_distance_based(distance, roi, localize_type)
#     elif is_correlation_based(localize_type):
#         lfactor = local_factor_correlation_based(correlation, localize_type)    
#     ind1 = np.where(lfactor>0)
#     return ind[ind1], lfactor[ind1]

def is_distance_based(localize_type):
    """Return True if the localization type is distance-based"""
    ltype = localize_type.split(',')
    if 'GC' in ltype:
        return True
    if 'step' in ltype:
        return True
    if 'exp' in ltype:
        return True
    return False

def is_correlation_based(localize_type):
    """Return True if the localization type is correlation-based"""
    ltype = localize_type.split(',')
    if 'SER' in ltype:
        return True
    if 'NICE' in ltype:
        return True
    return False

@njit(cache=True)
def local_factor_distance_based(dist, roi, localize_type='GC'):
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
    ltype = localize_type.split(',')

    if 'GC' in ltype: ##Gaspari-Cohn function (default)
        r = dist / (roi / 2)

        ind1 = np.where(r<1)
        loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
        lfactor[ind1] = loc1[ind1]

        ind2 = np.where(np.logical_and(r>=1, r<2))
        r[np.where(r==0)] = 1e-10  ##avoid divide by 0
        loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
        lfactor[ind2] = loc2[ind2]

    elif 'step' in ltype:  #step function from 1 to 0 at roi
        lfactor[np.where(dist<=roi)] = 1.0

    elif 'exp' in ltype:  ##exponential decay
        lfactor = np.exp(-dist/roi)

    ##otherwise set lfactor to 1
    lfactor[...] = 1.0
    return lfactor.reshape(shape)

@njit(cache=True)
def local_factor_correlation_based(covXY, varX, varY, fac=1):
    """
    localization based on correlation between variables
    algorithms include SER (Anderson 2016), NICE (Morzfeld et al. 2023)
    """
    # Ne = X.shape[1]
    ##lookup table
    # FileName = f'std_ro_Ne_{Ne}.mat'
    # dat = scipy.io.loadmat(f'matlab/std_ro_Ne_{Ne}.mat')
    # r = dat['r'].flatten()
    # stdCrs = dat['stdCrs'].flatten()
    # CorrXY = np.corrcoef(X, Y)[0:X.shape[0], X.shape[0]:]
    pass

#@njit(cache=True)
def interp1d(x, y, x_new):
    """
    A simple linear interpolation function for 1D arrays
    Replacing scipy.interpolate.interp1d, since it is not compatible with njit
    """
    y_new = np.empty_like(x_new)
    for i in range(len(x_new)):
        if x_new[i] <= x[0]:
            # extrapolate to the left
            y_new[i] = y[0] + (x_new[i] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        elif x_new[i] >= x[-1]:
            # extrapolate to the right
            y_new[i] = y[-1] + (x_new[i] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            # interpolate linearly
            for j in range(len(x) - 1):
                if x[j] <= x_new[i] <= x[j + 1]:
                    y_new[i] = y[j] + (y[j + 1] - y[j]) * (x_new[i] - x[j]) / (x[j + 1] - x[j])
                    break
    return y_new

@njit(cache=True)
def corrNICER(corr, r, stdCrs):
    pass
    # interp_func = interp1d(r, stdCrs, kind='linear', fill_value='extrapolate')
    # std_rho = interp_func(CorrXY)
    # std_rho[np.isclose(CorrXY, 1)] = 0
    # sig_rho = np.sqrt(np.sum(std_rho ** 2))

    # expo2 = 0
    # while True:
    #     expo2 += 2
    #     L = np.abs(CorrXY) ** expo2
    #     Corr_NICER = L * CorrXY
    #     if np.linalg.norm(Corr_NICER - CorrXY, 'fro') > fac * sig_rho:
    #         break

    # expo1 = expo2 - 2
    # rho_exp1 = CorrXY ** expo1
    # rho_exp2 = CorrXY ** expo2

    # al = np.arange(0.1, 1.1, 0.1)
    # PrevCorr = CorrXY
    # for kk in range(len(al)):
    #     L = (1 - al[kk]) * rho_exp1 + al[kk] * rho_exp2
    #     Corr_NICE = L * CorrXY
    #     if kk > 0 and np.linalg.norm(Corr_NICER - CorrXY, 'fro') > fac * sig_rho:
    #         Corr_NICE = PrevCorr
    #         break
    #     elif np.linalg.norm(Corr_NICE - CorrXY, 'fro') > fac * sig_rho:
    #         break
    #     PrevCorr = Corr_NICE
    # # Vy = np.diag(np.std(Y, axis=1))
    # # Vx = np.diag(np.std(X, axis=1))
    # # Cov_NICE = np.dot(Vx, np.dot(Corr_NICE, Vy))
    # lfactor = Corr_NICE / CorrXY
    # return lfactor

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


