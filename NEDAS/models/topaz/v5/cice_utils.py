import typing
import numpy as np
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var

def thickness_upper_limit(seaice_conc: np.ndarray, target_variable: typing.Literal['seaice','snow']) -> np.ndarray:
    """upper threshold for seaice_thick based on daily output of CICE 1993-2016"""
    upper_limit = np.full(seaice_conc.shape, 9999.)
    SIC0 = 0.55
    SIC1 = 0.8
    if target_variable == 'seaice':
        ind1 = np.where(np.logical_and(seaice_conc>SIC0, seaice_conc<SIC1))
        upper_limit[ind1] = 1.3 + np.exp(8.0*(seaice_conc[ind1]-0.76))
        ind2 = np.where(np.logical_and(seaice_conc>0, seaice_conc<SIC0))
        upper_limit[ind2] = 0.09 + 2.5*seaice_conc[ind2]

    elif target_variable == 'snow':
        ind1 = np.where(np.logical_and(seaice_conc>SIC0, seaice_conc<SIC1))
        upper_limit[ind1] = 0.2 + np.exp(2.0*(seaice_conc[ind1]-1.3))
        ind2 = np.where(np.logical_and(seaice_conc>0, seaice_conc<SIC0))
        upper_limit[ind2] = 0.03 + 0.7*seaice_conc[ind2]

    return upper_limit

def hi_cate(ncat, kcatb=0, kitd=1):
    """
    Defines ice thickness category boundaries based on the given scheme.
    
    Parameters:
        ncat (int): Number of ice thickness categories.
        kcatb (int): Determines the categorization scheme.
        kitd (int): Determines the remapping type (used when kcatb == 0).

    Returns:
        hilim (list of float): Ice thickness category boundaries.
    """
    hilim = np.zeros(ncat+1)  # Initialize the boundaries
    h_min = 0.01  # Default minimum ice thickness

    if kcatb == 0:
        if kitd == 1:  # Linear remapping
            cc1 = 3.0 / ncat
            cc2 = 15.0 * cc1
            cc3 = 3.0
            hilim[0] = 0.0
        else:  # Delta function remapping
            h_min = 0.1
            cc1 = max(1.1 / ncat, h_min)
            cc2 = 25.0 * cc1
            cc3 = 2.25
            hilim[0] = h_min

        for k in range(1, ncat + 1):
            x1 = (k - 2) / ncat
            hilim[k] = hilim[k - 1] + cc1 + cc2 * (1.0 + np.tanh(cc3 * (x1 - 1.0)))

    elif kcatb == 1:  # Linear scheme with additional coefficients
        cc1 = 3.0 / ncat
        cc2 = 0.5 / ncat
        hilim[0] = 0.0
        for k in range(1, ncat + 1):
            x1 = k - 1
            hilim[k] = x1 * (cc1 + (k - 2) * cc2)

    elif kcatb == 2:  # WMO standard categories
        if ncat == 5:
            hilim = [0.0, 0.3, 0.7, 1.2, 2.0, 999.0]
        elif ncat == 6:
            hilim = [0.0, 0.15, 0.3, 0.7, 1.2, 2.0, 999.0]
        elif ncat == 7:
            hilim = [0.0, 0.1, 0.15, 0.3, 0.7, 1.2, 2.0, 999.0]
        else:
            raise ValueError("kcatb=2 (WMO) must have ncat=5, 6, or 7")

    elif kcatb == -1:  # Single category
        hilim = [0.0, 100.0]

    else:  # Unsupported scheme
        raise ValueError("Error: no support for kcatb value in the current version!")

    return hilim

def fix_zsin_profile(Nlay, saltmax, depressT, nsal, msal):
    """
    Fixed salinity profile and melting temperature in ice layers
    only works under ktherm=1 for BL99 thermo
    """
    zSin = np.zeros(Nlay)
    Tmlt = np.zeros(Nlay)
    for k in range(Nlay-1):
        z = (k+0.5) / (Nlay-1)
        zSin[k] = 0.5*saltmax*(1-np.cos(np.pi*z**(nsal/(z+msal))))
        Tmlt[k] = -depressT*zSin[k]
    zSin[-1] = saltmax
    Tmlt[-1] = -depressT * saltmax
    return zSin, Tmlt

def adjust_ice_variables(prior_ice_file, post_ice_file,
                         fice, hice, mask,
                         aice_thresh, fice_thresh, hice_impact,
                         zSin, Tmlt) -> None:
    """
    Adjust the iced model restart file variables based on analysis of seaice properties: fice, hice
    """
    ncat = 5
    bkcat = hi_cate(ncat)

    aicen_f = nc_read_var(prior_ice_file, 'aicen')
    vicen_f = nc_read_var(prior_ice_file, 'vicen')
    vsnon_f = nc_read_var(prior_ice_file, 'vsnon')
    qsnon_f = nc_read_var(prior_ice_file, 'qsno001')

    ##bound aicen between 0,1
    aicen_f = np.maximum(np.minimum(aicen_f, 1), 0)

    aicen = aicen_f.copy()
    vicen = vicen_f.copy()
    vsnon = vsnon_f.copy()
    qicen_f = qsnon_f.copy()

    fice_f = np.sum(aicen_f, axis=0)
    aicen_f[np.where(np.logical_or(aicen_f <= aice_thresh, fice_f <= fice_thresh))] = 0.0
    fice_f = np.sum(aicen_f, axis=0)

    ##adjust for aicen, rescaled by analyzed fice
    ind = np.where(fice_f > fice_thresh)
    for k in range(ncat):
        aicen[k,...][ind] *= fice[ind] / fice_f[ind]
    ficem = np.sum(aicen, axis=0)
    ind = np.where(ficem > 1)
    for k in range(ncat):
        aicen[k,...][ind] /= ficem[ind]

    ind = np.where(fice_f <= fice_thresh)
    for k in range(ncat):
        aicen[k,...][ind] = 0.0
        vicen[k,...][ind] = 0.0
        vsnon[k,...][ind] = 0.0
    ficem = np.sum(aicen, axis=0)

    ind = np.where(np.logical_or(mask, fice <= fice_thresh))
    for k in range(ncat):
        aicen[k,...][ind] = 0.0
        vicen[k,...][ind] = 0.0
        vsnon[k,...][ind] = 0.0
        aicen_f[k,...][ind] = 0.0
        vicen_f[k,...][ind] = 0.0
    ficem[ind] = 0.0
    fice_f[ind] = 0.0

    ##adjust vicen
    ind = np.where(fice_f > fice_thresh)
    for k in range(ncat):
        ind1 = np.where(np.logical_and(aicen_f[k,...][ind] > 0, aicen[k,...][ind] > 0))
        vicen[k,...][ind][ind1] *= aicen[k,...][ind][ind1] / aicen_f[k,...][ind][ind1]
        #vsnon[k,...][ind][ind1] *= aicen[k,...][ind][ind1] / aicen_f[k,...][ind][ind1]

    ##when ice pack area assimilating hice
    ind1 = np.where(ficem[ind] > 0.75)
    sum_vice = np.sum(vicen, axis=0)[ind][ind1]
    Vtemp = ficem[ind][ind1] * (hice[ind][ind1]*hice_impact + sum_vice*(1-hice_impact)) / sum_vice
    for k in range(ncat):
        ind2 = np.where(aicen[k,...][ind][ind1] > 0)
        vicen[k,...][ind][ind1][ind2] *= Vtemp[ind2]
        ind2 = np.where(aicen[k,...][ind][ind1] <= 0)
        vicen[k,...][ind][ind1][ind2] = 0.0

    ##bound check again
    for k in range(ncat):
        ind2 = np.where(aicen[k,...][ind][ind1] > 0)
        vicen_min = aicen[k,...][ind][ind1][ind2] * bkcat[k]
        vicen_max = aicen[k,...][ind][ind1][ind2] * bkcat[k+1]
        vicen[k,...][ind][ind1][ind2] = np.maximum(np.minimum(vicen[k,...][ind][ind1][ind2], vicen_max), vicen_min)
        ind2 = np.where(aicen[k,...][ind][ind1] <= 0)
        vicen[k,...][ind][ind1][ind2] = 0.0
        vsnon[k,...][ind][ind1][ind2] = 0.0
        aicen[k,...][ind][ind1][ind2] = 0.0

    ind = np.where(fice_f <= fice_thresh)
    for k in range(ncat):
        vicen[k,...][ind] = 0.0
        vsnon[k,...][ind] = 0.0

    ind = np.where(np.logical_or(mask, ficem <= 0))
    for k in range(ncat):
        vicen[k,...][ind] = 0.0
        vsnon[k,...][ind] = 0.0
        aicen[k,...][ind] = 0.0
    ficem[ind] = 0.0

    ##replace the variables in posterior ice restart file
    ny, nx = fice.shape
    dims2 = {'nj':ny, 'ni':nx}
    dims3 = {'ncat':5, 'nj':ny, 'ni':nx}
    nc_write_var(post_ice_file, dims3, 'aicen', aicen, dtype=np.float64)
    nc_write_var(post_ice_file, dims3, 'vicen', vicen, dtype=np.float64)
    nc_write_var(post_ice_file, dims3, 'vsnon', vsnon, dtype=np.float64)

    ##other affected 2D variables
    ind = np.where(ficem <= 0)
    varlist_2d = ['uvel', 'vvel', 'strocnxT', 'strocnyT', 'frz_onset', 'iceumask']
    for i in range(4):
        varlist_2d.append(f'stressp_{i+1}')
        varlist_2d.append(f'stressm_{i+1}')
        varlist_2d.append(f'stress12_{i+1}')
    for vname in varlist_2d:
        var = nc_read_var(post_ice_file, vname)
        var[ind] = 0.0
        nc_write_var(post_ice_file, dims2, vname, var, dtype=np.float64)

    ##other affected 3D variables
    varlist_3d = ['iage', 'FY', 'alvl', 'vlvl', 'apnd', 'hpnd', 'ipnd', 'dhs', 'ffrac', 'Tsfcn']
    for i in range(7):
        varlist_3d.append(f'sice{i+1:03}')
        varlist_3d.append(f'qice{i+1:03}')
    varlist_3d.append('qsno001')
    for vname in varlist_3d:
        var = nc_read_var(post_ice_file, vname)
        for k in range(ncat):
            ind = np.where(np.logical_or(aicen[k,...] <= 0, ficem <= 0))
            if vname == 'Tsfcn':
                var[k,...][ind] = -1.8
                var[k,...][mask] = 0.0
            else:
                var[k,...][ind] = 0.0

            ind = np.where(aicen[k,...] > 0)
            if vname[:4] == 'sice':
                l = int(vname[-3:])
                var[k,...][ind] = zSin[l-1]

            #ind = np.where(aicen[k,...] <= aice_thresh)
            #if vname[:4] == 'qice':
            #    l = int(vname[-3:])
            #    Tmz = Tmlt[l-1]
            #    tmp1 = Tf[ind]+1.8
            #    Ti = -1.8 + tmp1*(l-0.5)/7
            #    var[k,...][ind] = -rhoi * (cp_ice*(Tmaz-Ti) + Lfresh*(1-Tmz/Ti) - cp_ocn*Tmz) / 7

        nc_write_var(post_ice_file, dims3, vname, var, dtype=np.float64)

