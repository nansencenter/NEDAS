from __future__ import annotations
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.config import Config

class Covariance:
    """
    Background error covariance model: a weighted blend of the dynamic (forecast) ensemble
    covariance and a static covariance sampled by an ensemble from a climatological bank,

        P = (1-beta) * P_dynamic + beta * alpha * P_static

    (Hamill & Snyder 2000; Wang et al. 2007; Counillon et al. 2009).

    beta=0 (default) is the pure ensemble covariance; beta=1 uses only the static covariance.
    The nens_static static members are a separate batch from the nens dynamic ones, with their
    own mem_id 0...nens_static-1 (Context.mem_list_static), and only enter the analysis.

    Args:
        nens (int): dynamic ensemble size
        beta (float): weight of the static covariance, in [0, 1]
        alpha (float): amplitude scale of the static covariance
        nens_static (int): number of static ensemble members
        hybrid_perturbation (bool): the mean is always updated with P; the dynamic perturbations
            are updated with the dynamic ensemble covariance alone if False (Wang et al. 2007),
            or with P (reduced Kalman gain, Counillon et al. 2009) if True
    """
    def __init__(self, nens: int, beta: float=0.0, alpha: float=1.0, nens_static: int=0,
                 hybrid_perturbation: bool=False):
        if not 0 <= beta <= 1:
            raise ValueError(f"covariance_def: beta={beta} is outside [0, 1]")
        if alpha <= 0:
            raise ValueError(f"covariance_def: alpha={alpha} should be positive")
        if nens_static < 0:
            raise ValueError(f"covariance_def: nens_static={nens_static} should be >= 0")
        if beta > 0 and nens_static < 2:
            raise ValueError(f"covariance_def: beta={beta} > 0 needs nens_static >= 2, got {nens_static}")
        if nens_static > 0 and beta < 1 and nens < 2:
            raise ValueError(f"covariance_def: beta={beta} < 1 needs nens >= 2 dynamic members, got {nens}")
        self.nens = nens
        self.beta = beta
        self.alpha = alpha
        self.nens_static = nens_static
        self.hybrid_perturbation = bool(hybrid_perturbation)

    def anomaly_factors(self) -> tuple[float, float]:
        """
        Scaling (fac_dynamic, fac_static) of the dynamic and static anomalies A_d, A_s (each
        about its own mean), so that Z = [fac_dynamic*A_d, fac_static*A_s] gives Z Z^T = P
        """
        fac_dynamic = math.sqrt(1 - self.beta) / math.sqrt(max(self.nens - 1, 1))
        fac_static = math.sqrt(self.beta * self.alpha) / math.sqrt(max(self.nens_static - 1, 1))
        return fac_dynamic, fac_static

def get_covariance(config: Config) -> Covariance:
    """Get the Covariance instance from config.covariance_def"""
    covariance_def = dict(config.covariance_def or {})
    # 'type' and 'config_file' are left from the old registry design, where 'ensemble' was the
    # only model; drop them so that existing config files still load
    covariance_type = covariance_def.pop('type', 'ensemble')
    covariance_def.pop('config_file', None)
    if covariance_type != 'ensemble':
        raise NotImplementedError(f"covariance_def: type '{covariance_type}' is not supported, "
                                  "set beta/alpha/nens_static for a static or hybrid covariance")
    return Covariance(config.nens, **covariance_def)

__all__ = ['Covariance', 'get_covariance']
