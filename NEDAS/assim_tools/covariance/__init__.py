from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.config import Config

class Covariance:
    """
    Background error covariance model: a weighted blend of the dynamic (forecast) ensemble
    covariance and a static covariance sampled by an ensemble from a climatological bank,

        P = (1-beta) * P_dynamic + beta * static_var_scaling * P_static

    (Hamill & Snyder 2000; Wang et al. 2007; Counillon et al. 2009).

    beta=0 (default) is the pure ensemble covariance; beta=1 uses only the static covariance.
    The nens_static static members are a separate batch from the nens dynamic ones, with their
    own mem_id 0...nens_static-1 (Context.mem_list_static), and only enter the analysis.

    Args:
        nens (int): dynamic ensemble size
        beta (float): weight of the static covariance, in [0, 1] (the alpha of Wang et al. 2007, Eq. 1;
            named beta as the covariance weights in variational hybrids)
        static_var_scaling (float): scaling of the static covariance, reducing the climatological
            variance of the static members to a background-error level (the alpha of EnOI, Evensen 2003)
        nens_static (int): number of static ensemble members
        static_dir (str): the bank of static members, restart files named as the model's own
            member files, read with io tag 'static' in both io modes (IOBackend.static_member_kwargs)
        hybrid_perturbation (bool): the mean is always updated with P; the dynamic perturbations
            are updated with the dynamic ensemble covariance alone if False (Wang et al. 2007),
            or with P (reduced Kalman gain, Counillon et al. 2009) if True
    """
    def __init__(self, nens: int, beta: float=0.0, static_var_scaling: float=1.0, nens_static: int=0,
                 hybrid_perturbation: bool=False, static_dir: str|None=None):
        if not 0 <= beta <= 1:
            raise ValueError(f"covariance_def: beta={beta} is outside [0, 1]")
        if static_var_scaling <= 0:
            raise ValueError(f"covariance_def: static_var_scaling={static_var_scaling} should be positive")
        if nens_static < 0:
            raise ValueError(f"covariance_def: nens_static={nens_static} should be >= 0")
        if beta > 0 and nens_static < 2:
            raise ValueError(f"covariance_def: beta={beta} > 0 needs nens_static >= 2, got {nens_static}")
        if nens_static > 0 and beta < 1 and nens < 2:
            raise ValueError(f"covariance_def: beta={beta} < 1 needs nens >= 2 dynamic members, got {nens}")
        self.nens = nens
        self.beta = beta
        self.static_var_scaling = static_var_scaling
        self.nens_static = nens_static
        self.hybrid_perturbation = bool(hybrid_perturbation)
        self.static_dir = static_dir
        # (time, source member) of each static member's restart file in static_dir,
        # set by get_covariance from covariance_def.static_list
        self.static_members: list[tuple[datetime, int|None]] = []

    def anomaly_factors(self) -> tuple[float, float]:
        """
        Scaling (fac_dynamic, fac_static) of the dynamic and static anomalies A_d, A_s (each
        about its own mean), so that Z = [fac_dynamic*A_d, fac_static*A_s] gives Z Z^T = P
        """
        fac_dynamic = math.sqrt(1 - self.beta) / math.sqrt(max(self.nens - 1, 1))
        fac_static = math.sqrt(self.beta * self.static_var_scaling) / math.sqrt(max(self.nens_static - 1, 1))
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
                                  "set beta/static_var_scaling/nens_static for a static or hybrid covariance")
    static_list = covariance_def.pop('static_list', None)
    covariance = Covariance(config.nens, **covariance_def)

    if covariance.nens_static > 0:
        if not covariance.static_dir or not static_list:
            raise ValueError("covariance_def: nens_static > 0 needs static_dir and static_list")
        covariance.static_members = read_static_list(static_list, covariance.nens_static)
    return covariance

def read_static_list(static_list: str, nens_static: int) -> list[tuple[datetime, int|None]]:
    """
    Read the list of static members from the file static_list: one line per member,
    '<time> [<member>]', the time of its restart file in the bank and the source member index
    (0-based as mem_id, omitted for a file without member suffix). Blank lines and '#' comments
    are skipped, the first nens_static members are used.
    """
    static_members = []
    with open(static_list) as f:
        for line in f:
            items = line.split('#')[0].split()
            if not items:
                continue
            time = datetime.fromisoformat(items[0])
            if time.tzinfo is None:
                time = time.replace(tzinfo=timezone.utc)
            member = int(items[1]) if len(items) > 1 else None
            static_members.append((time, member))
    if len(static_members) < nens_static:
        raise ValueError(f"covariance_def: static_list '{static_list}' lists {len(static_members)} members, "
                         f"fewer than nens_static={nens_static}")
    return static_members[:nens_static]

__all__ = ['Covariance', 'get_covariance', 'read_static_list']
