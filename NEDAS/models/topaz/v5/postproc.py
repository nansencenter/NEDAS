import numpy as np
from NEDAS.utils.njit import njit

@njit
def adjust_dp(dp, depth, onem):
    """
    Adjusts the pressure layers (dp) to ensure no negative values 
    and recalculates layer thickness based on depth constraints.

    Parameters:
    dp (ndarray): 3D array of pressure layers (kdm, jdm, idm).
    depth (ndarray): 2D array of depths (jdm, idm).
    onem (float): Scaling factor.

    Returns:
    ndarray: Adjusted dp array.
    """
    kdm, jdm, idm = dp.shape
    dp_ = dp.copy()  # Backup of the original dp
    press = np.zeros(kdm + 1)

    for j in range(jdm):
        for i in range(idm):
            # Move negative layers to neighbouring layers (forward sweep)
            for k in range(kdm - 1):
                dp[k+1, j, i] += min(0.0, dp[k, j, i])
                dp[k, j, i] = max(dp[k, j, i], 0.0)

            # Adjust lowermost layers (backward sweep)
            for k in range(kdm-1, 2, -1):
                dp[k-1, j, i] += min(0.0, dp[k, j, i])
                dp[k, j, i] = max(dp[k, j, i], 0.0)

            # Recalculate pressure layers
            press[0] = 0.0
            for k in range(kdm):
                press[k+1] = press[k] + dp[k, j, i]
                press[k+1] = min(depth[j, i] * onem, press[k+1])
            press[kdm] = depth[j, i] * onem

            # Recalculate dp based on updated pressures
            for k in range(kdm):
                dp[k, j, i] = press[k+1] - press[k]

            # Fallback for invalid depths
            if depth[j, i] > 100000.0 or depth[j, i] < 1.0:
                dp[:, j, i] = dp_[:, j, i]
    return dp

def stmt_fns_constans(thflag:int):
    if thflag == 0:
        # coefficients for sigma-0 (based on Brydon & Sun fit)
        c1 = -1.36471e-01
        c2 = 4.68181e-02
        c3 = 8.07004e-01
        c4 = -7.45353e-03
        c5 = -2.94418e-03
        c6 = 3.43570e-05
        c7 = 3.48658e-05
        pref = 0.0
    elif thflag == 2:
        # coefficients for sigma-2 (based on Brydon & Sun fit)
        c1 = 9.77093e+00
        c2 = -2.26493e-02
        c3 = 7.89879e-01
        c4 = -6.43205e-03
        c5 = -2.62983e-03
        c6 = 2.75835e-05
        c7 = 3.15235e-05
        pref = 2000.e4
    elif thflag == 4:
        # coefficients for sigma-4 (based on Brydon & Sun fit)
        c1 = 1.92362e+01
        c2 = -8.82080e-02
        c3 = 7.73552e-01
        c4 = -5.46858e-03
        c5 = -2.31866e-03
        c6 = 2.11306e-05
        c7 = 2.82474e-05
        pref = 4000.e4
    else:
        raise ValueError(f"Unknown thflag: {thflag}")
    return c1,c2,c3,c4,c5,c6,c7,pref

def stmt_fns_sigma(thflag:int, temp, saln):
    """Adapted from stmt_fns_sigma in EnKF-MPI-TOPAZ/SSHFromState"""
    c1,c2,c3,c4,c5,c6,c7,pref = stmt_fns_constans(thflag)
    return c1 + c3 * saln + temp * (c2 + c5 * saln + temp * (c4 + c7 * saln + c6 * temp))

def stmt_fns_kappaf(thflag:int, temp, saln, pres, thref=1.0e-3):
    c1,c2,c3,c4,c5,c6,c7,pref = stmt_fns_constans(thflag)
    # coefficients for kappa^(theta)
    # new values (w.r.t. t-toff, s-soff, prs-poff) from Shan Sun, 2/8/01
    toff = 3.0
    soff = 35.0
    poff = 2000.e4
    prefo = pref - poff
    qt = -2.41701E-01
    qs = -1.05131E-01
    qtt = 4.03001E-03
    qst = 1.06829E-03
    qttt = -3.03869E-05
    qpt = 1.00638E-09
    qpst = 1.48599E-11
    qptt = -1.31384E-11
    qthref = 1.0 / thref

    t = np.maximum(-2.0, np.minimum(32.0, temp)) - toff
    s = np.maximum(30.0, np.minimum(38.0, saln)) - soff
    prs = pres - poff

    kappaf = (1.0e-11 * qthref) * (prs - prefo) * (
              s * (qs + t * qst) +
              t * (qt + t * (qtt + t * qttt) +
              0.5 * (prs + prefo) * (qpt + s * qpst + t * qptt)) )
    return kappaf
