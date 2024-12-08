from typing import Literal
import numpy as np

def thickness_upper_limit(seaice_conc: np.ndarray, target_variable: Literal['seaice','snow']) -> np.ndarray:
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
