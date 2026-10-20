"""
Probit transforms of ensembles (DART's probit_transform_mod and bnrh_distribution_mod, v11.25).

A transform is fitted to each row of an ensemble array (nrow, nens) and kept as a dict p of
per-row arrays, so the same rows can be transformed back later (DART's distribution_params_type).
probit_given transforms other values with an already fitted p (DART's use_input_p).
"""
import numpy as np
from scipy.special import ndtr, ndtri, gammainc, gammaincinv, betainc, betaincinv
from NEDAS.utils.njit import njit

DISTRIBUTIONS = ('NORMAL_DISTRIBUTION', 'BOUNDED_NORMAL_RH_DISTRIBUTION', 'GAMMA_DISTRIBUTION',
                 'BETA_DISTRIBUTION', 'LOG_NORMAL_DISTRIBUTION', 'UNIFORM_DISTRIBUTION', 'KDE_DISTRIBUTION')
MAX_QUANTILE = 0.999999999999999   # DART normal_distribution_mod
UNIFORM_THRESHOLD = 0.01           # DART bnrh_distribution_mod: below it a bounded tail is uniform

@njit
def index_sort(x):
    """
    DART sort_mod index_sort (a heap sort): the order of x. Tied members come out in DART's order,
    which matters where the posterior is handed out by rank (the rank histogram filters).
    """
    num = x.size
    indx = np.arange(1, num + 1)       # 1-based, as in the Fortran
    if num <= 1:
        return indx - 1
    level = num // 2 + 1
    ind = num
    while True:
        if level > 1:
            level -= 1
            l_val_indx = indx[level-1]
            l_val = x[l_val_indx-1]
        else:
            l_val_indx = indx[ind-1]
            l_val = x[l_val_indx-1]
            indx[ind-1] = indx[0]
            ind -= 1
            if ind == 1:
                indx[0] = l_val_indx
                return indx - 1
        i = level
        j = 2 * level
        while j <= ind:
            if j < ind and x[indx[j-1]-1] < x[indx[j]-1]:
                j += 1
            if l_val < x[indx[j-1]-1]:
                indx[i-1] = indx[j-1]
                i = j
                j = 2 * j
            else:
                j = ind + 1
        indx[i-1] = l_val_indx


def to_probit_space(q):
    """DART inv_std_normal_cdf: the quantiles kept within [tiny, max_quantile]"""
    return ndtri(np.clip(q, np.finfo(float).tiny, MAX_QUANTILE))

def normal_cdf(x, mean, sd):
    return ndtr((x - mean) / sd)

def inv_weighted_normal_cdf(alpha, mean, sd, q):
    return mean + sd * to_probit_space(q / alpha)


def to_probit(ens, dist, bounded_below=False, bounded_above=False, lower_bound=0.0, upper_bound=1.0):
    """
    Fit the transform dist to each row of ens (nrow, nens), return the rows in probit space and p.

    p['ok'] is False for rows that cannot be transformed (no spread for the fitted distributions,
    DART's ierr); those are returned unchanged and should not be updated (DART skips them).
    """
    ens = np.asarray(ens, dtype=np.float64)
    p = {'dist': dist, 'ok': np.ones(ens.shape[0], dtype=bool)}
    if dist == 'NORMAL_DISTRIBUTION':
        return ens.copy(), p
    if dist == 'LOG_NORMAL_DISTRIBUTION':
        return np.log(ens), p
    if dist == 'UNIFORM_DISTRIBUTION':
        p.update(lower_bound=lower_bound, upper_bound=upper_bound)
        return probit_given(ens, p), p

    if dist == 'KDE_DISTRIBUTION':
        # ponytail: a Python loop over rows (fit and bisection per row); vectorize over rows if it is the cost
        fits = np.empty(ens.shape[0], dtype=object)
        fits[:] = [kde_row_fit(row, bounded_below, bounded_above, lower_bound, upper_bound) for row in ens]
        p.update(kde=fits, ok=np.array([f is not None for f in fits], dtype=bool))
        out = ens.copy()
        for r in np.where(p['ok'])[0]:
            out[r] = to_probit_space(kde_quantiles(ens[r], fits[r]))
        return out, p

    mean = ens.mean(axis=1)
    var = ens.var(axis=1, ddof=1)
    p['ok'] = var > 0
    if dist == 'GAMMA_DISTRIBUTION':
        # ponytail: rows without spread are flagged, where DART's method of moments divides by 0
        with np.errstate(divide='ignore', invalid='ignore'):
            p.update(shape=mean**2 / var, scale=var / mean)
    elif dist == 'BETA_DISTRIBUTION':
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = mean**2 * (1 - mean) / var - mean
            p.update(alpha=alpha, beta=alpha * (1 / mean - 1))
    elif dist == 'BOUNDED_NORMAL_RH_DISTRIBUTION':
        p.update(bnrh_fit(ens, bounded_below, bounded_above, lower_bound, upper_bound))
    else:
        raise NotImplementedError(f"QCEF: probit distribution {dist} (one of {DISTRIBUTIONS})")

    out = ens.copy()
    ok = p['ok']
    if dist == 'BOUNDED_NORMAL_RH_DISTRIBUTION':
        q = bnrh_ens_quantiles(ens[ok], p['sort'][ok], bounded_below, bounded_above, lower_bound, upper_bound)
        out[ok] = to_probit_space(q)
    else:
        out[ok] = probit_given(ens[ok], rows(p, ok))
    return out, p


def probit_given(x, p):
    """Values x (nrow, n) in probit space, by the transform p fitted to the same rows (use_input_p)"""
    x = np.asarray(x, dtype=np.float64)
    dist = p['dist']
    if dist == 'NORMAL_DISTRIBUTION':
        return x.copy()
    if dist == 'LOG_NORMAL_DISTRIBUTION':
        return np.log(x)
    if dist == 'UNIFORM_DISTRIBUTION':
        return to_probit_space((x - p['lower_bound']) / (p['upper_bound'] - p['lower_bound']))
    if dist == 'GAMMA_DISTRIBUTION':
        return to_probit_space(gammainc(p['shape'][:, None], np.maximum(x, 0) / p['scale'][:, None]))
    if dist == 'BETA_DISTRIBUTION':
        return to_probit_space(betainc(p['alpha'][:, None], p['beta'][:, None], np.clip(x, 0, 1)))
    if dist == 'BOUNDED_NORMAL_RH_DISTRIBUTION':
        return to_probit_space(np.array([bnrh_cdf_given(x[r], rows(p, r)) for r in range(x.shape[0])]))
    if dist == 'KDE_DISTRIBUTION':
        return to_probit_space(np.array([kde_quantiles(x[r], p['kde'][r]) for r in range(x.shape[0])]))
    raise NotImplementedError(f"QCEF: probit distribution {dist}")


def from_probit(probit, p):
    """Rows (nrow, nens) back from probit space by their fitted p; rows not ok are returned as they are"""
    probit = np.asarray(probit, dtype=np.float64)
    dist = p['dist']
    if dist == 'NORMAL_DISTRIBUTION':
        return probit.copy()
    if dist == 'LOG_NORMAL_DISTRIBUTION':
        return np.exp(probit)
    q = ndtr(probit)
    if dist == 'UNIFORM_DISTRIBUTION':
        return p['lower_bound'] + q * (p['upper_bound'] - p['lower_bound'])
    out = probit.copy()
    ok = p['ok']
    if dist == 'GAMMA_DISTRIBUTION':
        out[ok] = gammaincinv(p['shape'][ok, None], q[ok]) * p['scale'][ok, None]
    elif dist == 'BETA_DISTRIBUTION':
        out[ok] = betaincinv(p['alpha'][ok, None], p['beta'][ok, None], q[ok])
    elif dist == 'BOUNDED_NORMAL_RH_DISTRIBUTION':
        out[ok] = inv_bnrh_cdf(q[ok], rows(p, ok))
    elif dist == 'KDE_DISTRIBUTION':
        for r in np.where(ok)[0]:
            out[r] = inv_kde_quantiles(q[r], p['kde'][r])
    else:
        raise NotImplementedError(f"QCEF: probit distribution {dist}")
    return out


def rows(p, i):
    """The transform of the rows i of p"""
    return {k: (v[i] if isinstance(v, np.ndarray) else v) for k, v in p.items()}


# ---------------------------------------------------------------- bounded normal rank histogram

def bnrh_fit(ens, bounded_below, bounded_above, lower_bound, upper_bound):
    """DART bnrh_cdf: the sorted members and the normal tails of each row"""
    nens = ens.shape[1]
    sd = ens.std(axis=1, ddof=1)
    sort = np.sort(ens, axis=1)
    ok = sd > 0
    if bounded_below and np.any(sort[ok, 0] < lower_bound):
        raise ValueError(f"QCEF: ensemble member below the lower bound {lower_bound}")
    if bounded_above and np.any(sort[ok, -1] > upper_bound):
        raise ValueError(f"QCEF: ensemble member above the upper bound {upper_bound}")
    del_q = 1.0 / (nens + 1)
    dist_for_unit_sd = -to_probit_space(del_q)
    p = {'ok': ok, 'sort': sort, 'sd': sd,
         'mean_left': sort[:, 0] + dist_for_unit_sd * sd, 'mean_right': sort[:, -1] - dist_for_unit_sd * sd,
         'amp_left': np.ones(len(sd)), 'amp_right': np.ones(len(sd)),
         'uniform_left': np.zeros(len(sd), dtype=bool), 'uniform_right': np.zeros(len(sd), dtype=bool),
         'bounded_below': bounded_below, 'bounded_above': bounded_above,
         'lower_bound': lower_bound, 'upper_bound': upper_bound}
    with np.errstate(divide='ignore', invalid='ignore'):
        if bounded_below:
            bq = normal_cdf(lower_bound, p['mean_left'], sd)
            p['uniform_left'] = (del_q - bq) / del_q < UNIFORM_THRESHOLD
            p['amp_left'] = np.where(p['uniform_left'], 1.0, del_q / (del_q - bq))
        if bounded_above:
            bq = normal_cdf(upper_bound, p['mean_right'], sd)
            p['uniform_right'] = (bq - (1 - del_q)) / del_q < UNIFORM_THRESHOLD
            p['amp_right'] = np.where(p['uniform_right'], 1.0, del_q / (del_q - (1 - bq)))
    return p


def bnrh_ens_quantiles(ens, sort, bounded_below, bounded_above, lower_bound, upper_bound):
    """
    DART ens_quantiles, for the members in their own order: the j-th of n sorted members has
    quantile j/(n+1); tied members share the middle of their run, and members on a bound share
    half their run's mass next to it.
    """
    nrow, n = sort.shape
    j = np.arange(n)
    first = np.ones((nrow, n), dtype=bool)
    first[:, 1:] = sort[:, 1:] != sort[:, :-1]
    last = np.ones((nrow, n), dtype=bool)
    last[:, :-1] = first[:, 1:]
    start = np.maximum.accumulate(np.where(first, j, 0), axis=1)                    # run of ties: first
    end = np.minimum.accumulate(np.where(last, j, n - 1)[:, ::-1], axis=1)[:, ::-1]  # and last member
    q = (start + 1) / (n + 1) + (end - start) / (2 * (n + 1))
    if bounded_below:
        at = sort == lower_bound
        dups = at.sum(axis=1, keepdims=True)
        q = np.where(at, dups / (2 * (n + 1)), q)
    if bounded_above:
        at = sort == upper_bound
        dups = at.sum(axis=1, keepdims=True)
        q = np.where(at, 1 - dups / (2 * (n + 1)), q)
    # back to the members' order
    order = np.argsort(ens, axis=1, kind='stable')
    out = np.empty_like(q)
    np.put_along_axis(out, order, q, axis=1)
    return out


def bnrh_cdf_given(x, p):
    """DART bnrh_cdf_initialized for the values x (1-D) under the transform p of one row"""
    s, n = p['sort'], p['sort'].size
    lb, ub = p['lower_bound'], p['upper_bound']
    del_q = 1.0 / (n + 1)
    q_ens = bnrh_ens_quantiles(s[None], s[None], p['bounded_below'], p['bounded_above'], lb, ub)[0]
    if p['bounded_below'] and np.any(x < lb):
        raise ValueError(f"QCEF: value below the lower bound {lb}")
    if p['bounded_above'] and np.any(x > ub):
        raise ValueError(f"QCEF: value above the upper bound {ub}")
    k = np.searchsorted(s, x, side='left')      # s[k-1] < x <= s[k]
    q = np.empty(x.size)
    for i in range(x.size):
        xi, ki = x[i], k[i]
        if xi < s[0]:
            if p['uniform_left']:
                qi = (xi - lb) / (s[0] - lb) * del_q
            elif p['bounded_below']:
                qi = p['amp_left'] * (normal_cdf(xi, p['mean_left'], p['sd']) - normal_cdf(lb, p['mean_left'], p['sd']))
            else:
                qi = normal_cdf(xi, p['mean_left'], p['sd']) / normal_cdf(s[0], p['mean_left'], p['sd']) * del_q
            qi = qi if p['uniform_left'] else min(qi, q_ens[0])
        elif xi > s[-1]:
            if p['uniform_right']:
                qi = n * del_q + (xi - s[-1]) / (ub - s[-1]) * del_q
            else:
                q_last = normal_cdf(s[-1], p['mean_right'], p['sd'])
                if p['bounded_above']:
                    amp = p['amp_right']
                    upper_q = amp * normal_cdf(ub, p['mean_right'], p['sd'])
                    fract = (amp * normal_cdf(xi, p['mean_right'], p['sd']) - amp * q_last) / (upper_q - amp * q_last)
                else:
                    fract = (normal_cdf(xi, p['mean_right'], p['sd']) - q_last) / (1 - q_last)
                qi = min(n * del_q + fract * del_q, 1.0)
        elif xi == s[ki]:
            qi = q_ens[ki]
        else:
            qi = ki * del_q + (xi - s[ki-1]) / (s[ki] - s[ki-1]) * del_q
        q[i] = qi
    return q


def inv_bnrh_cdf(q, p):
    """DART inv_bnrh_cdf: values of the quantiles q (nrow, nens) under the transforms p of the rows"""
    s = p['sort']
    nrow, n = q.shape
    lb, ub = p['lower_bound'], p['upper_bound']
    del_q = 1.0 / (n + 1)
    sd, ml, mr = p['sd'][:, None], p['mean_left'][:, None], p['mean_right'][:, None]
    al, ar = p['amp_left'][:, None], p['amp_right'][:, None]
    region = np.clip(np.floor(q * (n + 1)).astype(int), 0, n)

    # interior: linear between the members region and region+1 (1-based)
    r = np.clip(region, 1, n - 1)
    lo, hi = np.take_along_axis(s, r - 1, axis=1), np.take_along_axis(s, r, axis=1)
    x = lo + (q - r * del_q) / del_q * (hi - lo)

    with np.errstate(divide='ignore', invalid='ignore'):
        # left tail
        lower_mass = al * normal_cdf(lb, ml, sd) if p['bounded_below'] else 0.0
        upper_mass = al * normal_cdf(s[:, :1], ml, sd)
        left = inv_weighted_normal_cdf(al, ml, sd, lower_mass + q / del_q * (upper_mass - lower_mass))
        if p['bounded_below']:
            left = np.where(p['uniform_left'][:, None], lb + q / del_q * (s[:, :1] - lb), left)
        # right tail
        upper_mass = ar * normal_cdf(ub, mr, sd) if p['bounded_above'] else 1.0
        lower_mass = ar * normal_cdf(s[:, -1:], mr, sd)
        fract = (q - n * del_q) / (1 - n * del_q)
        right = inv_weighted_normal_cdf(ar, mr, sd, lower_mass + fract * (upper_mass - lower_mass))
        if p['bounded_above']:
            right = np.where(p['uniform_right'][:, None], s[:, -1:] + fract * (ub - s[:, -1:]), right)

    x = np.where(region == 0, left, np.where(region == n, right, x))
    if p['bounded_below']:
        x = np.maximum(x, lb)
    if p['bounded_above']:
        x = np.minimum(x, ub)
    return x


# ---------------------------------------------------------------- kernel density estimate (DART kde_distribution_mod)
# Epanechnikov kernels with adaptive bandwidths (Grooms 2022, Grooms and Riedel 2024), boundary-corrected
# near a bound (Jones 1993); members on a bound are point masses there, the KDE is fitted to the rest.

# 9th-order (5-point) Gauss-Legendre quadrature, DART's default quadrature_order
_GQ_NODES = np.array([-np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3, -np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3, 0.0,
                      np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3, np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3])
_GQ_WEIGHTS = np.array([(322 - 13 * np.sqrt(70)) / 900, (322 + 13 * np.sqrt(70)) / 900, 128 / 225,
                        (322 + 13 * np.sqrt(70)) / 900, (322 - 13 * np.sqrt(70)) / 900])
_BISECTIONS = 64

def epanechnikov_kernel(x):
    return 0.75 * np.maximum(0.0, 1.0 - x**2)

def epanechnikov_cdf(x):
    x = np.clip(x, -1.0, 1.0)
    return np.clip(0.25 * (2.0 + 3.0 * x - x**3), 0.0, 1.0)

def boundary_correction(x):
    denom = (1 + x)**4 * (19 + 3 * x * (x - 6))
    return -64 * (-2 + x * (4 + 3 * x * (x - 2))) / denom, 240 * (x - 1)**2 / denom


def kde_bandwidths(s):
    """DART get_kde_bandwidths for the sorted members s: h0 scaled by the kth-nearest-neighbor density"""
    n = s.size
    h0 = 2.0 * s.std(ddof=1) / n**0.2
    k = int(np.floor(np.sqrt(n)))
    d = np.sort(np.abs(s[:, None] - s[None, :]), axis=1)
    d[d < 1e-3 * d.max(axis=1, keepdims=True)] = 0.0
    # the kth neighbor, or the next one at a nonzero distance
    pos = d[:, k:] > 0
    idx = np.where(pos.any(axis=1), k + pos.argmax(axis=1), n - 1)
    with np.errstate(divide='ignore'):
        f = 0.5 * idx / (n * d[np.arange(n), idx])
    f = f / f.max()
    g = np.exp(np.mean(np.log(f)))     # DART: product(f)**(1/n)
    return h0 * np.sqrt(g / f)


def kde_fit(ens, bounded_below, bounded_above, lower_bound, upper_bound) -> dict:
    """DART pack_kde_params, for the prior (uninformative likelihood): kernels, edges and the cdf there"""
    s = np.sort(ens)
    h = kde_bandwidths(s)
    n = s.size
    edges = np.sort(np.concatenate([s - h, s + h, [lower_bound if bounded_below else np.min(s - h)],
                                    [upper_bound if bounded_above else np.max(s + h)]]))
    k = {'s': s, 'h': h, 'edges': edges, 'norm': 1.0,
         'bounded_below': bounded_below, 'bounded_above': bounded_above,
         'lower_bound': lower_bound, 'upper_bound': upper_bound}
    # far enough from a bound, the unbounded estimate (cheaper) is the same
    if bounded_below and np.min(s - 2 * h) > lower_bound:
        k['bounded_below'], k['lower_bound'] = False, np.min(s - h)
    if bounded_above and np.max(s + 2 * h) < upper_bound:
        k['bounded_above'], k['upper_bound'] = False, np.max(s + h)
    a, b = edges[:-1], edges[1:]
    empty = (b <= a) | (k['bounded_below'] & (b <= k['lower_bound'])) | (k['bounded_above'] & (b > k['upper_bound']))
    mass = np.where(empty, 0.0, kde_gq(a, b, k))
    cdfs = np.concatenate([[0.0], np.cumsum(mass)])
    k['norm'] = cdfs.max()
    k['cdfs'] = cdfs / k['norm']
    return k


def kde_pdf(x, k):
    """DART kde_pdf at the values x (any shape)"""
    x = np.asarray(x, dtype=np.float64)
    s, h = k['s'], k['h']
    u = (x[..., None] - s) / h
    lx_lo, mx_lo, lx_hi, mx_hi = 1.0, 0.0, 1.0, 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        if k['bounded_below']:
            ul = (x[..., None] - k['lower_bound']) / h
            near = (ul >= 0) & (ul <= 1)
            lx, mx = boundary_correction(np.where(near, ul, 0.5))
            lx_lo, mx_lo = np.where(near, lx, 1.0), np.where(near, mx, 0.0)
        if k['bounded_above']:
            uu = (k['upper_bound'] - x[..., None]) / h
            near = (uu >= 0) & (uu <= 1)
            lx, mx = boundary_correction(np.where(near, uu, 0.5))
            lx_hi, mx_hi = np.where(near, lx, 1.0), np.where(near, mx, 0.0)
    pdf = np.sum((lx_lo + mx_lo * u) * (lx_hi - mx_hi * u) * epanechnikov_kernel(u) / h, axis=-1)
    pdf = np.maximum(pdf, 0.0) / (s.size * k['norm'])
    if k['bounded_below']:
        pdf = np.where(x <= k['lower_bound'], 0.0, pdf)
    if k['bounded_above']:
        pdf = np.where(x >= k['upper_bound'], 0.0, pdf)
    return pdf


def kde_gq(a, b, k):
    """Gauss quadrature of the pdf from a to b (arrays of the same shape)"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    xi = 0.5 * ((b - a)[..., None] * _GQ_NODES + (a + b)[..., None])
    return 0.5 * (b - a) * np.sum(_GQ_WEIGHTS * kde_pdf(xi, k), axis=-1)


def kde_cdf(x, k):
    """DART kde_cdf_params at the values x (1-D)"""
    x = np.asarray(x, dtype=np.float64)
    if not k['bounded_below'] and not k['bounded_above']:
        return np.sum(epanechnikov_cdf((x[:, None] - k['s']) / k['h']), axis=1) / k['s'].size
    edges = k['edges']
    j = np.maximum(np.searchsorted(edges, x, side='left'), 1)   # first interval edges[j-1] <= x <= edges[j]
    inside = j < edges.size
    j = np.minimum(j, edges.size - 1)
    q = np.clip(k['cdfs'][j-1] + kde_gq(edges[j-1], np.minimum(x, edges[j]), k), 0.0, 1.0)
    return np.where(x < edges[0], 0.0, np.where(inside, q, 1.0))


def inv_kde_cdf(u, k):
    """
    The values with the cdf u (1-D), by bisection within the support of the kernels
    (DART inv_cdf uses a secant/ITP search, the same root to its tolerance)
    """
    # ponytail: fixed-count vectorized bisection; a secant/ITP search as DART's if this is the cost
    u = np.minimum(np.asarray(u, dtype=np.float64), MAX_QUANTILE)
    lo = k['edges'][0] if not k['bounded_below'] else max(k['edges'][0], k['lower_bound'])
    hi = k['edges'][-1] if not k['bounded_above'] else min(k['edges'][-1], k['upper_bound'])
    lo, hi = np.full(u.shape, lo), np.full(u.shape, hi)
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        below = kde_cdf(mid, k) < u
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    x = 0.5 * (lo + hi)
    if k['bounded_below']:
        x = np.where(u <= 0, k['lower_bound'], x)
    if k['bounded_above']:
        x = np.where(u >= 1, k['upper_bound'], x)
    return x


def kde_row_fit(ens, bounded_below, bounded_above, lower_bound, upper_bound):
    """
    DART to_probit_kde, the fitted part: the fractions of members on each bound (point masses) and
    the KDE of the members between them; None for a row without spread
    """
    if np.max(np.abs(ens - ens[0])) <= 0:
        return None
    on_lo = (ens <= lower_bound) if bounded_below else np.zeros(ens.size, dtype=bool)
    on_hi = (ens >= upper_bound) if bounded_above else np.zeros(ens.size, dtype=bool)
    interior = ens[~(on_lo | on_hi)]
    f = {'q_lower': on_lo.mean(), 'q_upper': on_hi.mean(), 'interior0': interior[0] if interior.size else np.nan,
         'bounded_below': bounded_below, 'bounded_above': bounded_above,
         'lower_bound': lower_bound, 'upper_bound': upper_bound, 'kde': None}
    f['q_int'] = 1.0 - f['q_lower'] - f['q_upper']
    if interior.size > 1 and np.max(np.abs(interior - interior[0])) > 0:
        f['kde'] = kde_fit(interior, bounded_below, bounded_above, lower_bound, upper_bound)
    return f


def kde_quantiles(x, f):
    """Quantiles of the values x (1-D) in the fitted row f"""
    if f['kde'] is None:
        u = np.full(x.shape, 0.5 * f['q_int'] + f['q_lower'])
    else:
        u = f['q_int'] * kde_cdf(x, f['kde']) + f['q_lower']
    if f['bounded_below']:
        u = np.where(x <= f['lower_bound'], f['q_lower'], u)
    if f['bounded_above']:
        u = np.where(x >= f['upper_bound'], 1.0 - f['q_upper'], u)
    return u


def inv_kde_quantiles(u, f):
    """DART from_probit_kde: the values of the quantiles u (1-D) in the fitted row f"""
    if f['kde'] is None:
        x = np.full(u.shape, f['interior0'])
    else:
        x = inv_kde_cdf((u - f['q_lower']) / f['q_int'], f['kde'])
    if f['bounded_below']:
        x = np.where(u <= f['q_lower'], f['lower_bound'], x)
    if f['bounded_above']:
        x = np.where(u >= 1.0 - f['q_upper'], f['upper_bound'], x)
    return x
