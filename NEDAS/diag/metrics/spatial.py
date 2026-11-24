# deterministic field-vs-field verification metrics (spatial patterns),
# as opposed to metrics that need the full ensemble distribution
import numpy as np


def rmse(fld, fld_tr):
    """
    RMSE of a field against the truth field.

    Args:
        fld:    np.ndarray, shape (...) — field to verify (e.g. an ensemble
                mean, already reduced -- this metric has no use for the
                full ensemble beyond its mean)
        fld_tr: np.ndarray, shape (...) — truth / reference field

    Returns:
        float: spatially averaged RMSE
    """
    return float(np.sqrt(np.nanmean((fld - fld_tr) ** 2)))


def pattern_corr(fld, fld_tr):
    """
    Anomaly pattern correlation between a field and the truth.

    Args:
        fld:    np.ndarray, shape (...) — field to verify
        fld_tr: np.ndarray, shape (...) — truth / reference field

    Returns:
        float: Pearson correlation coefficient in [-1, 1]
    """
    a = fld.ravel()
    b = fld_tr.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask] - np.mean(a[mask])
    b = b[mask] - np.mean(b[mask])
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)
