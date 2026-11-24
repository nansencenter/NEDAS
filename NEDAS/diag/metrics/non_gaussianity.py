"""
Non-Gaussianity diagnostic metrics for ensemble forecasts.

Measures of how far the marginal distribution of an ensemble (at each grid
point) departs from a Gaussian, for the "measured forecast non-Gaussianity"
axis of the method cross-comparison (see the project note on non-Gaussian DA
cross-comparison):

- skewness          -- 3rd standardized moment (asymmetry; sign-bearing)
- excess_kurtosis   -- 4th standardized moment minus 3 (tailedness)
- negentropy        -- KL divergence of the sample density from the best-fit
                       Gaussian N(mu, sigma^2), i.e. entropy(Gauss) - entropy(sample)

All functions act on ensemble arrays of shape (nens, ...) and return arrays of
shape (...) (one value per grid point / degree of freedom), so they compose
with masked fields: NaNs are expected only in whole masked columns (the NEDAS
mask convention) and propagate to NaN in the metric at those points; a
spatially degenerate (zero-variance) column yields 0, not NaN.

Estimators (numpy only, no density fitting):
- Standardized moments use biased (sample) moments about the mean.
- negentropy estimates differential entropy with the Vasicek (1976)
  spacings estimator, H = mean(log(n/(2m) (x_(i+m) - x_(i-m)))), with
  m = floor(sqrt(nens)) and out-of-range order statistics clamped to the
  extremes. Designed for nens >= ~20; small ensembles give noisy estimates,
  which are clipped at 0 (true KL >= 0, a negative value is estimator noise).
"""

import numpy as np

__all__ = ['skewness', 'excess_kurtosis', 'negentropy', 'summarize_non_gaussianity']


def _guard_nens(fld_ens: np.ndarray, minimum: int) -> int:
    nens = fld_ens.shape[0]
    if nens < minimum:
        raise ValueError(
            f"non-Gaussianity metrics need nens >= {minimum} along axis 0, got {nens}")
    return nens


def skewness(fld_ens: np.ndarray) -> np.ndarray:
    """
    Sample skewness of the ensemble at each point (3rd standardized moment).

    g1 = m3 / m2^(3/2), with m_k the k-th moment about the ensemble mean
    (biased). Positive for right-skewed, negative for left-skewed; 0 at
    zero-variance points.

    Args:
        fld_ens: np.ndarray, shape (nens, ...) — ensemble members

    Returns:
        np.ndarray, shape (...): pointwise skewness in [-inf, inf]
    """
    _guard_nens(fld_ens, 2)
    dev = fld_ens - np.mean(fld_ens, axis=0)
    m2 = np.mean(dev ** 2, axis=0)
    m3 = np.mean(dev ** 3, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = m3 / m2 ** 1.5
    return np.where(m2 == 0, 0.0, raw)


def excess_kurtosis(fld_ens: np.ndarray) -> np.ndarray:
    """
    Excess kurtosis of the ensemble at each point.

    g2 = m4 / m2^2 - 3 (biased moments). > 0 heavy-tailed / peaked, < 0
    platykurtic (e.g. bimodal); 0 at zero-variance points.

    Args:
        fld_ens: np.ndarray, shape (nens, ...) — ensemble members

    Returns:
        np.ndarray, shape (...): pointwise excess kurtosis in [-2, inf]
    """
    _guard_nens(fld_ens, 2)
    dev = fld_ens - np.mean(fld_ens, axis=0)
    m2 = np.mean(dev ** 2, axis=0)
    m4 = np.mean(dev ** 4, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = m4 / m2 ** 2 - 3.0
    return np.where(m2 == 0, 0.0, raw)


def negentropy(fld_ens: np.ndarray) -> np.ndarray:
    """
    Negentropy = KL(sample density || N(mu, sigma^2)) at each point.

    Estimated as entropy of the best-fit Gaussian minus the Vasicek (1976)
    spacings estimate of the sample differential entropy. ~0 for a Gaussian
    ensemble, > 0 for any departure (skew or kurtosis); negative estimator
    noise is clipped to 0 (true KL >= 0).

    Args:
        fld_ens: np.ndarray, shape (nens, ...) — ensemble members (nens >= 3)

    Returns:
        np.ndarray, shape (...): pointwise negentropy in nats, >= 0
    """
    nens = _guard_nens(fld_ens, 3)
    sigma2 = np.var(fld_ens, axis=0, ddof=1)  # NaN in fully-masked columns
    out = np.full(fld_ens.shape[1:], np.nan)

    # degenerate points: best-fit Gaussian has zero entropy difference
    const = sigma2 == 0
    out[const] = 0.0

    live = ~const
    if not np.any(live):
        return out

    # Vasicek spacings entropy on the live points
    x = np.sort(fld_ens, axis=0)  # NaN columns sort to the end; fine, they stay NaN
    m = max(1, int(np.sqrt(nens)))
    i = np.arange(nens)
    ip = np.minimum(i + m, nens - 1)
    im = np.maximum(i - m, 0)
    spacing = x[ip, ...] - x[im, ...]  # 1-D index arrays -> shape (nens, ...)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log(nens * spacing / (2.0 * m))
    log_term = np.where(np.isfinite(log_term), log_term, np.nan)
    # mean over members, but only over finite spacings (avoids nanmean's
    # empty-slice warning on fully-masked columns)
    finite = np.isfinite(log_term)
    n_finite = np.sum(finite, axis=0)
    with np.errstate(invalid='ignore'):
        h_sample = np.sum(np.where(finite, log_term, 0.0), axis=0) / np.maximum(n_finite, 1)
    h_sample = np.where(n_finite > 0, h_sample, np.nan)

    # entropy of the best-fit Gaussian, 0.5*log(2 pi e sigma^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_gauss = 0.5 * np.log(2.0 * np.pi * np.e * sigma2)

    est = h_gauss - h_sample
    est = np.where(np.isfinite(est) & (est < 0), 0.0, est)
    out[live] = est[live]
    return out


def summarize_non_gaussianity(fld_ens: np.ndarray) -> dict[str, float]:
    """
    Domain-mean scalar summaries of the three metrics (for the method
    cross-comparison's y-axis / per-variable-per-cycle values).

    Spatial means use nanmean over the trailing axes; skewness magnitude is
    averaged as |skewness| so left- and right-skewed regions do not cancel.

    Args:
        fld_ens: np.ndarray, shape (nens, ...) — ensemble members

    Returns:
        dict with keys 'skewness' (mean |g1|), 'excess_kurtosis' (mean g2),
        'negentropy' (mean KL to best-fit Gaussian, nats)
    """
    def _mean_ax(arr: np.ndarray) -> float:
        return float(np.nanmean(arr))

    return {
        'skewness': _mean_ax(np.abs(skewness(fld_ens))),
        'excess_kurtosis': _mean_ax(excess_kurtosis(fld_ens)),
        'negentropy': _mean_ax(negentropy(fld_ens)),
    }
