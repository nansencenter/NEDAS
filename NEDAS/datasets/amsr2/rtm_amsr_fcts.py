#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""---------------------------------------------------------------------------
AMSR Ocean Algorithm; Frank J. Wentz, Thomas Meissner; Remote
Sensing Systems; Version 2; November 2; 2000.
Tb=f(V,W,L,Ts,Ti,c_ice)
V: columnar water vapor [mm]
W: windspeed over water [m/s]
L: columnar cloud liquid water [mm]
Ts: sea surface temperature [K]
Ti_ansrx: ice effective temperature [K]
c_ice: ice concentration [0-1]
e_icex: ice emissivity
-------------------------------------------------------------------------------"""
import numpy as np
import xarray as xr
import cmath

frequencies = np.array([6.93, 10.65, 18.70, 23.80, 36.50, 50.30, 52.80, 89.00])

########################### SEA ICE EMISSIVITIES ###########################
### 1-Original values from github RTM
#sea ice emissivities (same as in rtm_ssmi.py for 19v, 19h, 22v, 22h, 37v, 37h)
# checked with Rasmus (04.05.2015)
# 6GHz values from Leif (03.11.2015)
# freq: 6.93, 10.65, 18.70, 23.80, 36.50, 50.30, 52.80, 89.00

Eice_v = [0.96, 0.9, 0.95, 0.95, 0.93, 0.9, 0.9, 0.90]
Eice_h = [0.88, 0.9, 0.90, 0.90, 0.88, 0.9, 0.9, 0.83]
#Eice_v = ['-', '-', 0.95, '-', 0.93, '-', '-', '-']
#Eice_h = ['-', '-', 0.90, '-', 0.88, '-', '-', '-']
#['tb19v', 'tb19h', 'tb37v', 'tb37h']
#[1.0012614  0.92405898 0.92682112 0.86746858]
#Eice_v = ['-', '-', 0.99, '-', 0.927, '-', '-', '-']
#Eice_h = ['-', '-', 0.924, '-', 0.868, '-', '-', '-']

######
### 2-Values from ASI3 algorithm Lu Junshen for February (no values for 50.30 and 52.80 fqs)
# FYI values
#Eice_v = [0.951, 0.952, 0.965, 0.963, 0.944, '', '', 0.882]
#Eice_h = [0.852,  0.857,  0.851,  0.867,  0.845, '', '', 0.814]
# MYI values
#Eice_v = [0.962, 0.939, 0.896, 0.860, 0.774, '', '', 0.801]
#Eice_h = [0.862, 0.845, 0.817, 0.785, 0.707, '', '', 0.754]
#############################################################################

# Channel dependent mixing of Ts and 272K
#   (from PVASR SICCI1 SICCI_PVASR_D2.5_(SIC)_Issue\ 1.0.pdf page 152-153)
# Checked with Leif (04.05.2015). The PVASR values are only entered for
#    the 19v, 19h, 22v, 22h, 37v, 37h and 90v, 90h channels. The other
#    frequencies (not used in OSISAF/SICCI) keep a default 0.4
# 6GHz values from Leif (03.11.2015)
Tmix_v = [0.45, 0.4, 0.75, 0.90, 0.95, 0.4, 0.4, 0.97]
Tmix_h = [0.40, 0.4, 0.47, 0.60, 0.70, 0.4, 0.4, 0.97]

# main coefficient tables
b0 = np.array([239.50E+0,  239.51E+0,  240.24E+0,  241.69E+0,  239.45E+0,  242.10E+0,  245.87E+0,  242.58E+0])
b1 = np.array([213.92E-2,  225.19E-2,  298.88E-2,  310.32E-2,  254.41E-2,  229.17E-2,  250.61E-2,  302.33E-2])
b2 = np.array([-460.60E-4, -446.86E-4, -725.93E-4, -814.29E-4, -512.84E-4, -508.05E-4, -627.89E-4, -749.76E-4])
b3 = np.array([457.11E-6,  391.82E-6,  814.50E-6,  998.93E-6,  452.02E-6,  536.90E-6,  759.62E-6,  880.66E-6])
b4 = np.array([-16.84E-7,  -12.20E-7,  -36.07E-7,  -48.37E-7,  -14.36E-7,  -22.07E-7,  -36.06E-7,  -40.88E-7])
b5 = np.array([0.50E+0,     0.54E+0,    0.61E+0,    0.20E+0,    0.58E+0,    0.52E+0,    0.53E+0,    0.62E+0])
b6 = np.array([-0.11E+0,   -0.12E+0,   -0.16E+0,   -0.20E+0,   -0.57E+0,   -4.59E+0,  -12.52E+0,   -0.57E+0])
b7 = np.array([-0.21E-2,   -0.34E-2,   -1.69E-2,   -5.21E-2,   -2.38E-2,   -8.78E-2,  -23.26E-2,   -8.07E-2])
ao1 = np.array([8.34E-3,    9.08E-3,   12.15E-3,   15.75E-3,   40.06E-3,  353.72E-3, 1131.76E-3,   53.35E-3])
ao2 = np.array([-0.48E-4,  -0.47E-4,   -0.61E-4,   -0.87E-4,   -2.00E-4,  -13.79E-4,   -2.26E-4,   -1.18E-4])
av1 = np.array([0.07E-3,    0.18E-3,    1.73E-3,    5.14E-3,    1.88E-3,    2.91E-3,    3.17E-3,    8.78E-3])
av2 = np.array([0.00E-5,    0.00E-5,   -0.05E-5,    0.19E-5,    0.09E-5,    0.24E-5,    0.27E-5,    0.80E-5])

aL1 = np.array([0.0078, 0.0183, 0.0556, 0.0891,  0.2027,  0.3682,  0.4021,  0.9693])
aL2 = np.array([0.0303, 0.0298, 0.0288, 0.0281,  0.0261,  0.0236,  0.0231,  0.0146])
aL3 = np.array([0.0007, 0.0027, 0.0113, 0.0188,  0.0425,  0.0731,  0.0786,  0.1506])
aL4 = np.array([0.0000, 0.0060, 0.0040, 0.0020, -0.0020, -0.0020, -0.0020, -0.0020])
aL5 = np.array([1.2216, 1.1795, 1.0636, 1.0220,  0.9546,  0.8983,  0.8943,  0.7961])

r0v = np.array([-0.27E-3,  -0.32E-3,  -0.49E-3,  -0.63E-3,  -1.01E-3, -1.20E-3, -1.23E-03, -1.53E-3])
r0h = np.array([0.54E-3,   0.72E-3,   1.13E-3,   1.39E-3,   1.91E-3,  1.97E-3,  1.97E-03,  2.02E-3])
r1v = np.array([-0.21E-4,  -0.29E-4,  -0.53E-4,  -0.70E-4,  -1.05E-4, -1.12E-4, -1.13E-04, -1.16E-4])
r1h = np.array([0.32E-4,   0.44E-4,   0.70E-4,   0.85E-4,   1.12E-4,  1.18E-4,  1.19E-04,  1.30E-4])
r2v = np.array([-2.10E-5,  -2.10E-5,  -2.10E-5,  -2.10E-5,  -2.10E-5, -2.10E-5, -2.10E-05, -2.10E-5])
r2h = np.array([-25.26E-6, -28.94E-6, -36.90E-6, -41.95E-6, -54.51E-6, -5.50E-5, -5.50E-5,  -5.50E-5])
r3v = np.array([0.00E-6,   0.08E-6,   0.31E-6,   0.41E-6,   0.45E-6,  0.35E-6,  0.32E-06, -0.09E-6])
r3h = np.array([0.00E-6,  -0.02E-6,  -0.12E-6,  -0.20E-6,  -0.36E-6, -0.43E-6, -0.44E-06, -0.46E-6])

m1v = np.array([0.00020, 0.00020, 0.00140, 0.00178, 0.00257, 0.00260, 0.00260, 0.00260])
m1h = np.array([0.00200, 0.00200, 0.00293, 0.00308, 0.00329, 0.00330, 0.00330, 0.00330])
m2v = np.array([0.00690, 0.00690, 0.00736, 0.00730, 0.00701, 0.00700, 0.00700, 0.00700])
m2h = np.array([0.00600, 0.00600, 0.00656, 0.00660, 0.00660, 0.00660, 0.00660, 0.00660])

def _get_chn_idx(channel):
    """Get index of channel from name
    """

    try:
        chn_idx = {'06v'  : 0,  '06h' : 0,
                   '10v' : 1, '10h' : 1,
                   '19v' : 2, '19h' : 2,
                   '22v' : 3, '22h' : 3,
                   '37v' : 4, '37h' : 4,
                   '50v' : 5, '50h' : 5,
                   '52v' : 6, '52h' : 6,
                   '90v' : 7, '90h' : 7,}[channel.lower()]
    except KeyError:
        raise ValueError('Cannot do RTM for channel %s' % channel)

    return chn_idx

def _is_Vpol(channel):
    """Check if v-polarisation
    """

    return channel.endswith('v')

def calc_epsilon(Ts, channel, freq = None) :
    """
    Calculates the dielectric constant ε of sea water (epsilon)

    :Parameters:
    Ts : float, numpy array
        Surface temperature
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}

    :Returns:
    epsilon: float or numpy array
        Dielectric constant of sea water
    """

    # Get channel index and frequency
    i = _get_chn_idx(channel)
    if freq is None: freq = frequencies[i]
    #
    epsilon_R = 4.44 # Dielectric constant at inf. freq. This value is from wentz and meisner, 2000, p. 28
    s = 35.0 # Salinity in parts per thousand
    ny = 0.012 # Spread factor. Klein and Swift is using 0.02 which is giving a higher epsilon_R (4.9)
    light_speed = 3.00E10 # Speed of light, [cm/s]
    free_space_permittivity = 8.854E-12
    #eq 43
    epsilon_S = (87.90 * np.exp(-0.004585 * (Ts - 273.15))) * (np.exp(-3.45E-3 * s + 4.69E-6 * s**2 + 1.36E-5 * s * (Ts - 273.15)))
    #eq 44
    lambda_R = (3.30 * np.exp(-0.0346 * (Ts - 273.15) + 0.00017 * (Ts - 273.15)**2))-\
            (6.54E-3 * (1 - 3.06E-2 * (Ts - 273.15) + 2.0E-4 * (Ts - 273.15)**2) * s)
    #eq 41
    C = 0.5536 * s
    #eq 42
    delta_t = 25.0 - (Ts - 273.15)
    #eq 40
    qsi = 2.03E-2 + 1.27E-4 * delta_t + 2.46E-6 * delta_t**2 - C * (3.34E-5 - 4.60E-7 * delta_t + 4.60E-8 * delta_t**2)
    #eq 39
    sigma = 3.39E9 * (C**0.892) * np.exp(-delta_t * qsi)
    #
    llambda = (light_speed/(freq * 1E9))
    #eq 35
    #print('lambda_R:', lambda_R)
    #print('llambda:', llambda)
    #print('ny:', ny)
    epsilon = epsilon_R + ((epsilon_S - epsilon_R)/(1.0 + ((cmath.sqrt(-1) * lambda_R)/llambda)**(1.0 - ny))) - ((2.0 * cmath.sqrt(-1) * sigma * llambda)/light_speed)

    return epsilon

def calc_ocean_emissivity(W, Ts, theta, channel) :
    """
    Calculates emissivity of sea surface

    :Parameters:
    W : float, numpy array
        Windspeed over water
    Ts : float, numpy array
        Surface temperature
    theta: float
        Incidence angle
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}

    :Returns:
    emissivity: float or numpy array
        Sea surface emissivity
    """

    # Get channel index
    i = _get_chn_idx(channel)
    # Get the dielectric constant of sea water (epsilon)
    epsilon = calc_epsilon(Ts, channel)
    # Next equations split for each polarisation
    if _is_Vpol(channel) : # vertical polarisation
        #eq.45
        rho = (epsilon * np.cos(np.deg2rad(theta)) - np.sqrt(epsilon - np.sin(np.deg2rad(theta))**2))/\
             (epsilon * np.cos(np.deg2rad(theta)) + np.sqrt(epsilon - np.sin(np.deg2rad(theta))**2))
        C_R_0 = (4.887E-8 - 6.108E-8 * (Ts - 273.0)**3)
        r0 = r0v[i]; r1 = r1v[i]; r2 = r2v[i]; r3 = r3v[i];
        W_1 = 3.0; W_2 = 12.0
        f1 = m1v[i]; f2 = m2v[i]
    else: # horizontal polarisation
        #eq.45
        rho = (np.cos(np.deg2rad(theta)) - np.sqrt(epsilon - np.sin(np.deg2rad(theta))**2))/\
             (np.cos(np.deg2rad(theta)) + np.sqrt(epsilon - np.sin(np.deg2rad(theta))**2))
        C_R_0 = 0
        r0 = r0h[i]; r1 = r1h[i]; r2 = r2h[i]; r3 = r3h[i];
        W_1 = 7.0; W_2 = 12.0
        f1 = m1h[i]; f2 = m2h[i]
    #eq.46
    R_0 = np.absolute(rho)**2 + C_R_0
    #eq.57
    R_geo = R_0 - (r0 + r1 * (theta - 53.0) + r2 * (Ts - 288.0) + r3 * (theta - 53.0) * (Ts - 288.0)) * W
    #eq.60
    w_low = (W <= W_1)
    w_mid = (W_1 < W) * (W < W_2)
    w_high = (W >= W_2)
    F = w_low * (f1 * W) +\
                w_mid * (f1 * W + 0.5 * (f2 - f1) * ((W - W_1)**2)/(W_2 - W_1)) +\
                w_high * (f2 * W - 0.5 * (f2 - f1) * (W_2 + W_1))
    R = (1 - F) * R_geo
    ocean_emissivity = 1 - R

    return ocean_emissivity

def calc_emissivity(V, L, Ts, Tb, theta, channel) :
    """
    Calculates effective surface emissivity

    :Parameters:
    V : float, numpy array
        Columnar water vapor
    L : float, numpy array
        Columnar cloud liquid water
    Ts : float, numpy array
        Surface temperature
    Tb : float or numpy array
        Observed brightness temperature
    theta: float
        Incidence angle
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}

    :Returns:
    emissivity: float or numpy array
        Effective surface emissivity
    """

    _, TBD, _, TBU, tau = calc_down_up_welling(V, L, Ts, theta, channel)
    emissivity_value = (Tb - TBU - tau * TBD)/(tau * (Ts - TBD))

    return emissivity_value

def calc_transmittance(V, L, Ts, TD, theta, channel):
    """
    Calculates athmosphere transmittance

    :Parameters:
    V : float, numpy array
        Columnar water vapor
    L : float, numpy array
        Columnar cloud liquid water
    Ts : float, numpy array
        Surface temperature
    TD : float, numpy array
        Effective downwelling temperature
    theta: float
        Incidence angle
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}

    :Returns:
    tau: float or numpy array
        Athmosphere transmittance
    """

    # Get channel index
    i = _get_chn_idx(channel)
    Tl = (Ts + 273.0) / 2.0
    #eq 28
    AO = ao1[i] + ao2[i] * (TD - 270.0)
    #eq 29
    AV = av1[i] * V + av2[i] * V**2
    #eq 33
    AL = aL1[i] * (1.0 - aL2[i] * (Tl - 283.0)) * L
    #eq 22
    tau = np.exp((-1.0/np.cos(np.deg2rad(theta))) * (AO + AV + AL))

    return tau

def calc_down_up_welling(V, L, Ts, theta, channel) :
    """
    Calculates atmosphere up and downwelling temperatures

    :Parameters:
    V : float, numpy array
        Columnar water vapor
    L : float, numpy array
        Columnar cloud liquid water
    Ts : float, numpy array
        Surface temperature
    theta: float
        Incidence angle
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}

    :Returns:
    TD, TBD, TU, TBU, tau : floats or numpy arrays
        TD : Effective downwelling temperature
        TBD : Downwelling brightness temperature
        TU : Effective upwelling temperature
        TBU : Upwelling brightness temperature
        tau : Athmosphere transmittance
    """

    # Get channel index
    i = _get_chn_idx(channel)
    Tl = (Ts + 273.0) / 2.0
    #eq 27
    Tv = np.where(V <= 48, 273.16 + 0.8337 * V - 3.029E-5 * (V**3.33), 301.16)
    G = np.where(np.fabs(Ts - Tv) <= 20, 1.05 * (Ts - Tv) * (1 - ((Ts - Tv)**2)/1200.0), (Ts - Tv) * 14/np.fabs(Ts - Tv))
    ###
    #eq26
    TD = b0[i] + b1[i] * V + b2[i] * V**2 + b3[i] * V**3 + b4[i] * V**4 + b5[i] * G
    TU = TD + b6[i] + b7[i] * V
    #
    tau = calc_transmittance(V, L, Ts, TD, theta, channel)
    #eq 24
    TBU = TU * (1.0 - tau)
    TBD = TD * (1.0 - tau)

    return TD, TBD, TU, TBU, tau

def calc_omega(W, tau, channel, freq = None) :
    """
    Calculates correction factor for sea surface reflectance

    :Parameters:
    W : float, numpy array
        Windspeed over water
    tau: float or numpy array
        Athmosphere transmittance
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}

    :Returns:
    omega: float or numpy array
        Correction factor for sea surface reflectance
    """

    # Get channel index and frequency
    i = _get_chn_idx(channel)
    if freq is None: freq = frequencies[i]
    #
    if i >= 4: Delta_S2 = 5.22E-3 * W
    else: Delta_S2 = 5.22E-3 * (1 - 0.00748 * (37.0 - freq)**1.3) * W
    Delta_S2 = (Delta_S2 > 0.069) * 0.069 + (Delta_S2 <= 0.069) * Delta_S2 # set all values gt 0.069 to 0.069
    #eq.62
    term = Delta_S2 - 70.0 * Delta_S2**3
    # Next equations split for each polarisation
    if _is_Vpol(channel) : # vertical polarisation
        Omega = (2.5 + 0.018 * (37.0 - freq)) * term * tau**3.4
    else: # horizontal polarisation
        Omega = (6.2 - 0.001 * (37.0 - freq)**2) * term * tau**2.0

    return Omega

def observed_tb(V, W, L, Ts, ice_conc, theta, channel) :
    """
    Calculates brightness temperature as seen by the sensor over sea.
    No correction for wind direction.

    :Parameters:
    V : float, numpy array
        Columnar water vapor
    W : float, numpy array
        Windspeed over water
    L : float, numpy array
        Columnar cloud liquid water
    Ts : float, numpy array
        Surface temperature
    ice_conc : float, numpy array
        Sea ice concentration
    theta: float
        Incidence angle
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}

    :Returns:
    Tb : float or numpy array
        Observed brightness temperature

    """

    # Get channel index
    i = _get_chn_idx(channel)

    # Cosmic microwave background temperature
    T_C = 2.726

    # Ice concentration
    ice_conc = np.clip(ice_conc, 0., 1.)
    # Mix temperature and ice emissivity
    if _is_Vpol(channel) : tmix = Tmix_v[i]; e_ice = Eice_v[i]
    else: tmix = Tmix_h[i]; e_ice = Eice_h[i]
    # Ice temperature
    Ti = np.clip(tmix * Ts + (1. - tmix) * 272, 0, 272)

    # Get upwelling, downwelling temperatures and transmittance
    TD, TBD, _, TBU, tau = calc_down_up_welling(V, L, Ts, theta, channel)
    # Get reflection reductance from surface roughness
    Omega = calc_omega(W, tau, channel)
    # Get sea surface emissivity
    emissivity = calc_ocean_emissivity(W, Ts, theta, channel)

    #eq.61 sky radiation scattered upward by Earth surface
    T_BOmega = ((1 + Omega) * (1 - tau) * (TD - T_C) + T_C)

    # Calculate Tb
    Tb = TBU +  \
                tau * (
                (1.0 - ice_conc) * emissivity * Ts +
                ice_conc * e_ice * Ti +
                (1.0 - ice_conc) * (1.0 - emissivity) * (T_BOmega) +
                ice_conc * (1.0 - e_ice) * (TBD + tau * T_C))

    return Tb

def simulated_tb_v01(V, W, L, Ts, ice_conc, theta, channel) :
    """
    Simulates brightness temperature from constant ice and water emissivity values.
    No correction for wind direction.

    :Parameters:
    V : float, numpy array
        Columnar water vapor
    W : float, numpy array
        Windspeed over water
    L : float, numpy array
        Columnar cloud liquid water
    Ts : float, numpy array
        Surface temperature
    ice_conc : float, numpy array
        Sea ice concentration
    theta: float
        Incidence angle
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}

    :Returns:
    Tb : float or numpy array
        Simulated brightness temperature
    """

    # Get channel index
    i = _get_chn_idx(channel)
    #
    Ew_v = ['-', '-', 0.65, '-', 0.75, '-', '-', '-']
    Ew_h = ['-', '-', 0.33, '-', 0.41, '-', '-', '-']
    if _is_Vpol(channel) :
        e_ice = Eice_v[i]; e_water = Ew_v[i]
    else :
        e_ice = Eice_h[i]; e_water = Ew_h[i]
    #

    # Ice concentration
    ice_conc = np.clip(ice_conc, 0., 1.)

    # Compute effective emissivity
    effective_em = (1 - ice_conc) * e_water + ice_conc * e_ice

    # Get upwelling, downwelling temperatures and transmittance
    _, TBD, _, TBU, tau = calc_down_up_welling(V, L, Ts, theta, channel)

    # Compute Tb
    Tb = TBU + tau * (effective_em * Ts + TBD * (1 - effective_em))
    #Tb = TBU + tau*((1 - ice_conc)*e_water*Ts + ice_conc*e_ice*Ts + TBD(1 - (1 - ice_conc)*e_water + ice_conc*e_ice))
    #Tb = TBU + tau*((1.0 - ice_conc)*emissivity*Ts + ice_conc*e_ice*Ti + (1.0 - ice_conc)*\
    #(1.0 - emissivity)*(T_BOmega + tau*T_C) + ice_conc*(1.0 - e_ice)*(TBD + tau*T_C))
    return Tb

def simulated_tb_v02(V, W, L, Ts, ice_conc, theta, channel) :
    """
    Simulates brightness temperature from constant ice and computed water emissivity values.
    No correction for wind direction.

    :Parameters:
    V : float, numpy array
        Columnar water vapor
    W : float, numpy array
        Windspeed over water
    L : float, numpy array
        Columnar cloud liquid water
    Ts : float, numpy array
        Surface temperature
    ice_conc : float, numpy array
        Sea ice concentration
    theta: float
        Incidence angle
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}

    :Returns:
    Tb : float or numpy array
        Simulated brightness temperature

    """

    # Get channel index
    i = _get_chn_idx(channel)

    if _is_Vpol(channel) :
        e_ice = Eice_v[i];
    else :
        e_ice = Eice_h[i];
    #
    e_water = calc_ocean_emissivity(W, Ts, theta, channel)

    # Ice concentration
    ice_conc = np.clip(ice_conc, 0., 1.)

    # Compute effective emissivity
    effective_em = (1 - ice_conc) * e_water + ice_conc * e_ice

    # Get upwelling, downwelling temperatures and transmittance
    _, TBD, _, TBU, tau = calc_down_up_welling(V, L, Ts, theta, channel)

    # Compute Tb
    Tb = TBU + tau*(effective_em * Ts + TBD * (1 - effective_em))

    return Tb

def calc_emissivity_plan(x, y, channel, dict_coeffs) :
    """
    Computes emissivity (z) as a plan: z = a1*x + a2*y + c
    Coefficients a1, a2, c have been computed from TPD files with SIC = 1

    :Parameters:
    x, y : float, numpy array
            input variables of plane with x, y = T2M, DAL
    a1, a2 : float
            coefficients of equation of plane
    c : float
        value of intercept

    :Returns:
    z : float or numpy array
        Simulated emissivity
    """
    a1, a2 = dict_coeffs[channel]['a1'], dict_coeffs[channel]['a2']
    c = dict_coeffs[channel]['c']
    x1, y1 = x, y
    z = a1*x1 + a2*y1 + c

    return z

def simulated_tb_v03(V, W, L, Ts, ice_conc, theta, channel, ow_bias = 0, opt_em = 0, dal = None, dict_coeffs = {}, file_atlas = None, file_ml = None) :
    """
    Simulates brightness temperature from computed ice and water emissivity values.
    No correction for wind direction.

    :Parameters:
    V : float, numpy array
        Columnar water vapor
    W : float, numpy array
        Windspeed over water
    L : float, numpy array
        Columnar cloud liquid water
    Ts : float, numpy array
        Surface temperature
    ice_conc : float, numpy array
        Sea ice concentration
    theta: float
           Incidence angle
    channel : {'6v', '6h', '10v', '10h', '19v', '19h', '22v', '22h', '37v', '37h', '50v', '50h', '52v', '52h', '90v', '90h'}
    ow_bias : float, scalar
              open water bias (K) for Tbs
    opt_em : float, scalar
             0:'fix', 1:'dal', 2:'atlas'
    dal : float, numpy array
          Distance Along the Line (K)
    dict_coeffs : dictionnary, dict_coeffs[channel]['a1'], dict_coeffs[channel]['a2'], dict_coeffs[channel]['c']
                 Coefficients for computation of ice emissivity from DAL and Ts

    :Returns:
    Tb : float, numpy array
        Simulated brightness temperature
    """
    # Get channel index
    i = _get_chn_idx(channel)

    # Emissivity over sea ice (check option chosen: em_opt)
    ice_emissivity_options = {0 : 'fix', 1 : 'atlas', 2 : 'dal', 3 : 'ml'}
    if ice_emissivity_options[opt_em] == 'fix' :
        if _is_Vpol(channel) :
            e_ice = Eice_v[i]
        else :
            e_ice = Eice_h[i]
    elif ice_emissivity_options[opt_em] == 'atlas' :
        data_atlas = xr.open_dataset(file_atlas)
        e_ice = data_atlas[f'em_{channel}'][:].data
    elif ice_emissivity_options[opt_em] == 'dal' :
        e_ice = calc_emissivity_plan(Ts, dal, channel, dict_coeffs)
    elif ice_emissivity_options[opt_em] == 'ml' :
        e_ice = xr.open_dataset(file_ml)[f'Prediction_AMSR2_e{channel}'].data

    # Emissivity over water
    e_water = calc_ocean_emissivity(W, Ts, theta, channel)

    # Ice concentration
    ice_conc = np.clip(ice_conc, 0., 1.)

    # Compute surface emissivity
    surface_em = (1 - ice_conc) * e_water + ice_conc * e_ice

    # Get upwelling, downwelling temperatures and transmittance
    _, TBD, _, TBU, tau = calc_down_up_welling(V, L, Ts, theta, channel)

    # Compute Tb
    Tb = TBU + tau*(surface_em * Ts + TBD * (1 - surface_em))

    # OW bias correction
    # Use the same dims and coords for ice_conc
    # Apply mask and bias
    Tb = np.where(ice_conc >= 0.15, Tb, Tb - ow_bias)

    return Tb, e_ice
