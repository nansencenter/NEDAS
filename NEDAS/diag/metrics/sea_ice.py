"""
Sea ice concentration verification metrics.
"""
import numpy as np

__all__ = ['iiee']


def iiee(fld: np.ndarray, fld_tr: np.ndarray, threshold: float = 0.15,
         cell_area: float = 1.0) -> float:
    """
    Integrated Ice Edge Error (IIEE): total area where the binary ice/no-ice
    classification of a field disagrees with a reference, i.e. where one
    field says "ice" (concentration > threshold) and the other says
    "no ice", summed over the domain.

    IIEE = cell_area * sum( (fld > threshold) != (fld_tr > threshold) )

    Args:
        fld:       np.ndarray, shape (...) -- concentration field to verify
                   (fractional, 0-1), e.g. model or ensemble-mean SIC
        fld_tr:    np.ndarray, shape (...) -- reference/truth concentration
                   field, same shape and units as fld
        threshold: float -- ice/no-ice concentration threshold (default 0.15,
                   the standard sea-ice-extent convention)
        cell_area: float -- area of one grid cell, in whatever units the
                   caller wants IIEE reported in (e.g. km^2, or 1e-6*km^2 for
                   results in 10^6 km^2); default 1.0 returns a disagreement
                   pixel count instead of a physical area

    Returns:
        float: total disagreement area (or pixel count if cell_area=1.0).
        NaNs in either input mark invalid/masked cells, which are excluded
        from the comparison entirely (not counted as a disagreement even if
        the other field says "ice").
    """
    valid = np.isfinite(fld) & np.isfinite(fld_tr)
    ice = (fld > threshold) & valid
    ice_tr = (fld_tr > threshold) & valid
    return float(np.sum((ice != ice_tr) & valid) * cell_area)
