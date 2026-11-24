import numpy as np
from scipy.optimize import root_scalar
from scipy.special import gammainc, gammaincinv
from NEDAS.utils.njit import njit
from NEDAS.core import Context
from NEDAS.assim_tools.assimilators.serial import SerialAssimilator
from NEDAS.assim_tools.assimilators.EAKF.core import update_local_state_linear, update_local_obs_linear
from NEDAS.assim_tools.assimilators.QCEF import probit
from NEDAS.assim_tools.assimilators.QCEF.probit import normal_cdf, inv_weighted_normal_cdf, to_probit_space

FILTER_KINDS = ('EAKF', 'ENKF', 'UNBOUNDED_RHF', 'GAMMA_FILTER', 'BOUNDED_NORMAL_RHF', 'KDE_FILTER')


class QCEFAssimilator(SerialAssimilator):
    """
    Quantile-conserving ensemble filter framework (Anderson 2022, 2023), DART's filter_assim in Python.

    Each variable has a filter_kind, the obs-space update of the obs of that variable, and a probit
    distribution: the state and obs priors are mapped to probit space before the serial loop, the
    increments of each obs are regressed there and the result mapped back after. The settings are
    those of the DART assimilator (DART/default.yml), so a config runs with either.
    With EAKF and NORMAL_DISTRIBUTION everywhere this is the EAKF.
    """
    # from default.yml
    filter_kind: str
    probit_dist: str
    qceff: dict
    sort_obs_inc: bool
    rectangular_quadrature: bool
    gaussian_likelihood_tails: bool
    seed: int

    def option(self, name: str, v) -> dict:
        """filter_kind, dist and bounds of a variable (component v of a vector, or -1)"""
        opt = {'filter_kind': self.filter_kind, 'dist': self.probit_dist, 'bounded_below': False,
               'bounded_above': False, 'lower_bound': 0.0, 'upper_bound': 1.0}
        qceff = self.qceff or {}
        opt.update(qceff.get(name, {}))
        if v is not None and v >= 0:
            opt.update(qceff.get(f"{name}_{'xy'[v]}", {}))
        if opt['filter_kind'] not in FILTER_KINDS:
            raise NotImplementedError(f"QCEF: filter_kind {opt['filter_kind']} (one of {FILTER_KINDS})")
        return opt

    @staticmethod
    def bounds(opt: dict) -> tuple:
        return opt['bounded_below'], opt['bounded_above'], float(opt['lower_bound']), float(opt['upper_bound'])

    # ---------------------------------------------------------------- the serial loop hooks

    def assimilation_algorithm(self, c: Context):
        self.rng = np.random.default_rng(self.seed)   # the same draws on every rank (ENKF)
        self.obs_opt = {(r, v): self.option(rec.name, -1 if v is None else v)
                        for r, rec in c.obs.info.records.items() for v in ((0, 1) if rec.is_vector else (None,))}
        super().assimilation_algorithm(c)

    def transform_prior(self, c: Context, state_data: dict, obs_data: dict, X: np.ndarray, Y: np.ndarray) -> None:
        """State X (nloc, nfld, nens) and obs Y (nlobs, nens) priors to probit space, in place"""
        self.X0, self.Y0 = X.copy(), Y.copy()
        self.state_p = []
        for n, (rec_id, v) in enumerate(state_data['field_ids']):
            name = c.state.info.fields[rec_id].name if rec_id != 'scalar' else str(v)
            opt = self.option(name, v)
            X[:, n, :], p = probit.to_probit(X[:, n, :], opt['dist'], *self.bounds(opt))
            self.state_p.append(p)
        self.obs_p = []   # (rows, p) for each obs record and component
        for (r, v), opt in self.obs_opt.items():
            rows = np.where((obs_data['obs_rec_id'] == r) & (obs_data['obs_comp'] == (-1 if v is None else v)))[0]
            Y[rows], p = probit.to_probit(Y[rows], opt['dist'], *self.bounds(opt))
            self.obs_p.append((rows, p))
        self.obs_row_p = {}
        for rows, p in self.obs_p:
            for k, i in enumerate(rows):
                self.obs_row_p[i] = (p, k)

    def owned_obs_prior(self, i: int, prior: np.ndarray) -> np.ndarray:
        """The current prior of local obs i, back in physical space for its obs increment"""
        p, k = self.obs_row_p[i]
        if not p['ok'][k]:
            return prior
        return probit.from_probit(prior[None], probit.rows(p, [k]))[0]

    def obs_space_update(self, obs_rec_id, v, obs_p: dict):
        """Obs increments by the filter_kind of the obs, then the prior and increments in probit space"""
        opt = self.obs_opt[(obs_rec_id, v)]
        prior = obs_p['prior']
        incr = self.obs_increment(prior, obs_p['prior_static'], obs_p['obs'], obs_p['err_std'], opt)
        bounds = self.bounds(opt)
        prior_probit, p = probit.to_probit(prior[None], opt['dist'], *bounds)
        if not p['ok'][0]:
            return None, None    # DART: an obs whose prior cannot be transformed is not assimilated
        post_probit = probit.probit_given((prior + incr)[None], p)
        return prior_probit[0], post_probit[0] - prior_probit[0]

    def transform_post(self, c: Context, state_data: dict, obs_data: dict, X: np.ndarray, Y: np.ndarray) -> None:
        """Back to physical space; the rows that could not be transformed keep their prior (DART skips them)"""
        for n, p in enumerate(self.state_p):
            X[:, n, :] = probit.from_probit(X[:, n, :], p)
            X[~p['ok'], n, :] = self.X0[~p['ok'], n, :]
        for rows, p in self.obs_p:
            Y[rows] = probit.from_probit(Y[rows], p)
            Y[rows[~p['ok']]] = self.Y0[rows[~p['ok']]]
        del self.X0, self.Y0

    # ---------------------------------------------------------------- obs increment and regression

    def obs_increment(self, obs_prior, obs_prior_static, obs, obs_err, opt=None) -> np.ndarray:
        """DART obs_increment, for the filter_kind and bounds in opt (default: the global filter_kind)"""
        opt = opt or self.option('', -1)
        obs_var = obs_err**2
        prior_mean = np.mean(obs_prior)
        prior_var = np.var(obs_prior, ddof=1)
        if obs_var == 0 and prior_var == 0:
            raise ValueError("QCEF: obs error variance and prior spread are both 0")
        if obs_var == 0:
            return obs - obs_prior
        if prior_var == 0:
            return np.zeros_like(obs_prior)
        kind = opt['filter_kind']
        if kind == 'EAKF':
            return obs_increment_eakf(obs_prior, prior_mean, prior_var, obs, obs_var)
        if kind == 'ENKF':
            return obs_increment_enkf(obs_prior, prior_var, obs, obs_var, self.rng, self.sort_obs_inc)
        if kind == 'UNBOUNDED_RHF':
            return obs_increment_rank_histogram(obs_prior, prior_var, obs, obs_var,
                                                self.gaussian_likelihood_tails, self.rectangular_quadrature)
        if kind == 'GAMMA_FILTER':
            return obs_increment_gamma(obs_prior, prior_mean, prior_var, obs, obs_var)
        if kind == 'BOUNDED_NORMAL_RHF':
            return obs_increment_bounded_norm_rhf(obs_prior, obs, obs_var, *self.bounds(opt))
        if kind == 'KDE_FILTER':
            return obs_increment_qcef(obs_prior, obs, obs_err)
        raise NotImplementedError(f"QCEF: filter_kind {kind}")

    def update_local_state(self, state_prior, state_static, obs_prior, obs_prior_static, obs_incr,
                           ind, state_h_dist, state_v_dist, state_t_dist,
                           hroi, vroi, troi,
                           h_local_func, v_local_func, t_local_func, correlation_local_func,
                           impact_on_variable) -> None:
        # linear regression in probit space: the EAKF's, without static members
        update_local_state_linear(state_prior, state_static, obs_prior, obs_prior_static, obs_incr,
                                  ind, state_h_dist, state_v_dist, state_t_dist,
                                  hroi, vroi, troi,
                                  h_local_func, v_local_func, t_local_func, correlation_local_func,
                                  impact_on_variable, 1.0, 0.0, True)

    def update_local_obs(self, obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                         ind, h_dist, v_dist, t_dist,
                         hroi, vroi, troi,
                         h_local_func, v_local_func, t_local_func, correlation_local_func,
                         impact_on_variable) -> None:
        update_local_obs_linear(obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                                ind, h_dist, v_dist, t_dist,
                                hroi, vroi, troi,
                                h_local_func, v_local_func, t_local_func, correlation_local_func,
                                impact_on_variable, 1.0, 0.0, True)


# ---------------------------------------------------------------- obs-space increments (DART assim_tools_mod)

def obs_increment_eakf(ens, prior_mean, prior_var, obs, obs_var):
    var_ratio = obs_var / (prior_var + obs_var)
    new_mean = var_ratio * (prior_mean + prior_var * obs / obs_var)
    return np.sqrt(var_ratio) * (ens - prior_mean) + new_mean - ens


def obs_increment_enkf(ens, prior_var, obs, obs_var, rng, sort_obs_inc=False):
    """Perturbed obs, shifted to have the obs as their mean"""
    new_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    temp_obs = rng.normal(obs, np.sqrt(obs_var), ens.size)
    temp_obs += obs - temp_obs.mean()
    incr = new_var * (ens / prior_var + temp_obs / obs_var) - ens
    if sort_obs_inc:
        # the posterior members in the order of the prior ones, for the smallest increments
        new = ens + incr
        incr = np.empty_like(ens)
        incr[probit.index_sort(ens)] = new[probit.index_sort(new)] - np.sort(ens)
    return incr


def obs_increment_gamma(ens, prior_mean, prior_var, obs, obs_var):
    """Gamma prior times gamma likelihood, quantiles of the prior members kept in the posterior"""
    prior_shape, prior_scale = prior_mean**2 / prior_var, prior_var / prior_mean
    like_shape, like_scale = obs**2 / obs_var, obs_var / obs
    post_shape = prior_shape + like_shape - 1
    post_scale = prior_scale * like_scale / (prior_scale + like_scale)
    if post_shape <= 0:
        raise ValueError(f"QCEF: posterior gamma shape is not positive, {post_shape}")
    q = gammainc(prior_shape, np.maximum(ens, 0) / prior_scale)
    return gammaincinv(post_shape, q) * post_scale - ens


def obs_increment_rank_histogram(ens, prior_var, obs, obs_var, gaussian_likelihood_tails=False,
                                 rectangular_quadrature=True):
    """
    Rank histogram filter (Anderson 2010): 1/(n+1) prior mass between members and in each normal tail,
    times the likelihood, and the posterior members placed at the quantiles i/(n+1)
    """
    if not rectangular_quadrature:
        raise NotImplementedError("QCEF: UNBOUNDED_RHF with trapezoidal quadrature")
    n = ens.size
    e_ind = probit.index_sort(ens)
    x = ens[e_ind]
    like = np.exp(-(x - obs)**2 / (2 * obs_var)) / np.sqrt(2 * np.pi * obs_var)
    like_dense = (like[:-1] + like[1:]) / 2
    prior_sd = np.sqrt(prior_var)
    dist_for_unit_sd = -to_probit_space(1.0 / (n + 1))
    left_mean = x[0] + dist_for_unit_sd * prior_sd
    right_mean = x[-1] - dist_for_unit_sd * prior_sd

    mass = np.empty(n + 1)
    if gaussian_likelihood_tails:
        var_ratio = obs_var / (prior_var + obs_var)
        new_sd_left = new_sd_right = np.sqrt(var_ratio * prior_var)
        new_var = var_ratio * prior_var
        new_mean_left = var_ratio * (left_mean + prior_var * obs / obs_var)
        new_mean_right = var_ratio * (right_mean + prior_var * obs / obs_var)
        w = lambda m, nm: (np.exp(-0.5 * (m**2 / prior_var + obs**2 / obs_var - nm**2 / new_var))
                           / np.sqrt(prior_var + obs_var) / np.sqrt(2 * np.pi))
        prod_weight_left, prod_weight_right = w(left_mean, new_mean_left), w(right_mean, new_mean_right)
        mass[0] = normal_cdf(x[0], new_mean_left, new_sd_left) * prod_weight_left
        mass[n] = (1 - normal_cdf(x[-1], new_mean_right, new_sd_right)) * prod_weight_right
    else:
        new_sd_left = new_sd_right = prior_sd
        new_mean_left, new_mean_right = left_mean, right_mean
        prod_weight_left, prod_weight_right = like[0], like[-1]
        mass[0] = like[0] / (n + 1)
        mass[n] = like[-1] / (n + 1)
    mass[1:n] = like_dense / (n + 1)

    mass_sum = mass.sum()
    cumul = np.concatenate([[0.0], np.cumsum(mass / mass_sum)])
    left_amp, right_amp = prod_weight_left / mass_sum, prod_weight_right / mass_sum

    umass = np.arange(1, n + 1) / (n + 1)
    new = np.empty(n)
    left, right = umass < cumul[1], umass > cumul[n]
    new[left] = inv_weighted_normal_cdf(left_amp, new_mean_left, new_sd_left, umass[left])
    new[right] = 2 * new_mean_right - inv_weighted_normal_cdf(right_amp, new_mean_right, new_sd_right, 1 - umass[right])
    mid = ~(left | right)
    # the box j (between members j and j+1, 1-based) holding the mass: cumul[j] <= umass <= cumul[j+1]
    j = np.clip(np.searchsorted(cumul, umass[mid], side='left') - 1, 1, n - 1)
    new[mid] = x[j-1] + (umass[mid] - cumul[j]) / (cumul[j+1] - cumul[j]) * (x[j] - x[j-1])

    incr = np.empty(n)
    incr[e_ind] = new - x
    return incr


def truncated_normal_like(x, obs, obs_var, bounded_below, bounded_above, lower_bound, upper_bound):
    """DART get_truncated_normal_like: the obs error normal, renormalized to the bounds"""
    obs_sd = np.sqrt(obs_var)
    cdf_lo = normal_cdf(lower_bound, x, obs_sd) if bounded_below else 0.0
    cdf_hi = normal_cdf(upper_bound, x, obs_sd) if bounded_above else 1.0
    return np.exp(-(x - obs)**2 / (2 * obs_var)) / (cdf_hi - cdf_lo)


def obs_increment_bounded_norm_rhf(ens, obs, obs_var, bounded_below, bounded_above, lower_bound, upper_bound):
    """Bounded normal rank histogram prior (BNRH) times the truncated normal likelihood"""
    like = truncated_normal_like(ens, obs, obs_var, bounded_below, bounded_above, lower_bound, upper_bound)
    if not like.sum() > 0:
        return np.zeros_like(ens)    # the likelihood underflows: flat, no increments
    like = like / like.sum()
    p = probit.bnrh_fit(ens[None], bounded_below, bounded_above, lower_bound, upper_bound)
    q = probit.bnrh_ens_quantiles(ens[None], p['sort'], bounded_below, bounded_above, lower_bound, upper_bound)[0]
    sort_ind = probit.index_sort(ens)
    post = inv_bnrh_cdf_like(q, probit.rows(p, 0), like[sort_ind])
    return post - ens


def inv_bnrh_cdf_like(quantiles, p, like):
    """
    DART inv_bnrh_cdf_like: the values of the quantiles in the BNRH prior times the likelihood
    (given at the sorted members, normalized), the tails scaled by their likelihood at the outer members
    """
    s, n = p['sort'], p['sort'].size
    lb, ub = p['lower_bound'], p['upper_bound']
    sd, ml, mr = p['sd'], p['mean_left'], p['mean_right']
    del_q = 1.0 / (n + 1)
    like = like / (like.sum() + like[0] / 2 + like[-1] / 2)
    q = np.empty(n)
    q[0] = like[0]
    q[1:] = like[0] + np.cumsum((like[:-1] + like[1:]) / 2)

    x = np.empty(n)
    for i, cq in enumerate(quantiles):
        region = int(np.searchsorted(q, cq, side='right'))   # number of q <= cq: DART's first j with cq < q(j), less 1
        if region == 0:
            if p['bounded_below'] and p['uniform_left']:
                x[i] = lb + cq / q[0] * (s[0] - lb)
            else:
                amp = q[0] / del_q * p['amp_left']
                lower_mass = amp * normal_cdf(lb, ml, sd) if p['bounded_below'] else 0.0
                upper_mass = amp * normal_cdf(s[0], ml, sd)
                x[i] = inv_weighted_normal_cdf(amp, ml, sd, lower_mass + cq / q[0] * (upper_mass - lower_mass))
        elif region == n:
            fract = (cq - q[-1]) / (1 - q[-1])
            if p['bounded_above'] and p['uniform_right']:
                x[i] = s[-1] + fract * (ub - s[-1])
            else:
                amp = (1 - q[-1]) / del_q
                upper_mass = amp * p['amp_right'] * normal_cdf(ub, mr, sd) if p['bounded_above'] else amp
                lower_mass = amp * p['amp_right'] * normal_cdf(s[-1], mr, sd)
                x[i] = inv_weighted_normal_cdf(amp * p['amp_right'], mr, sd,
                                               lower_mass + fract * (upper_mass - lower_mass))
        else:
            x[i] = s[region-1] + (cq - q[region-1]) / (q[region] - q[region-1]) * (s[region] - s[region-1])
    if p['bounded_below']:
        x = np.maximum(x, lb)
    if p['bounded_above']:
        x = np.minimum(x, ub)
    return x


# ---------------------------------------------------------------- KDE obs increment (I. Grooms)
# Kernel density estimate prior times a Gaussian likelihood, quantiles conserved by root finding:
# KDE_FILTER. DART has its own KDE_FILTER (kde_distribution_mod); this one is not checked against it.

def obs_increment_qcef(obs_prior, obs, obs_err) -> np.ndarray:

    obs_increment = np.zeros_like(obs_prior)

    # If all members are equal, return zero increments.
    d_max = np.max(np.absolute(obs_prior[0] - obs_prior[:]))
    if (d_max <= 0.0):
        return obs_increment

    # Get prior and posterior distribution parameters. This computes the bandwidths twice, which is wasteful. TODO: fix
    params_prior = get_kde_params(obs_prior, 0.0, np.inf)
    params_post  = get_kde_params(obs_prior, obs, obs_err)

    for i in range(params_prior["nens"]):  # Loop iterations are independent, parallelizable
        u = kde_cdf(obs_prior[i], params_prior)
        f = lambda x: kde_cdf(x, params_post) - u
        fprime = lambda x: kde_pdf(x, params_post)
        # TODO: Add logic to catch errors. Maybe a better initial guess.
        obs_increment[i] = root_scalar(f, fprime=fprime, x0=obs_prior[i], bracket=(params_post["edges"][0],params_post["edges"][-1])).root - obs_prior[i]

    return obs_increment


@njit
def epanechnikov_kernel(x) -> np.ndarray:
    return 0.75 * np.maximum(0.0, 1.0 - x**2)


@njit
def epanechnikov_cdf(x) -> np.ndarray:
    x_truncated = np.minimum(1.0, np.maximum(-1.0, x))
    return 0.25 * (2.0 + 3.0 * x_truncated - x_truncated**3)


@njit
def get_kde_bandwidths(obs_prior: np.ndarray) -> np.ndarray:
    ## uses the observation-space forecast ensemble to compute kernel bandwidths
    nens = obs_prior.shape[0]
    d_max = np.absolute(obs_prior[0] - obs_prior[:]).max()
    #if (d_max <= 0.0): There should be a warning/error here. Can't continue if all ensemble members have the same value.
    ens_mean = np.mean(obs_prior)
    ens_sd   = np.std(obs_prior)
    h0 = 2.0 * ens_sd / (nens**0.2)  # This would be the kernel width if the widths were not adaptive.
                                         # It would be better to use min(sd, iqr/1.34) but don't want to compute iqr
    k = int(np.floor( np.sqrt(nens) ))  # distance to kth nearest neighbor is used to set bandwidth for k defined here
    f_tilde = np.zeros(nens)
    for i in range(nens):
        ## This loop can fail if the kth nearest neighbor has distance 0, in which case you need to
        ## search for the first neighbor that has a nonzero distance.
        dist  = np.sort(np.absolute(obs_prior[i] - obs_prior[:]))
        d_max = dist.max()
        # dist  = np.where(dist <= 1.E-3 * d_max, 0.0, dist)  # replace small distances with 0; only used with checks that aren't implemented yet
        f_tilde[i] = 0.5 * (k - 1) / (nens * dist[k])       # Initial density estimate
    f_tilde[:] = f_tilde[:] / np.max(f_tilde[:])            # Avoids overflow in the next line
    g = f_tilde.prod()**(1.0 / nens)
    lamda = np.sqrt(g / f_tilde)
    return h0 * lamda


def gauss_quad(a, b, f):
    ## Apply three-point (fifth-order) Gauss-Legendre quadrature to integrate f(x) from x=a to x=b

    # Ideally we would compute these once, rather than every time that the function is called.
    gq_nodes   = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    gq_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    # Local nodes & weights
    nodes   = 0.5 * ((b - a) * gq_nodes + a + b)
    weights = 0.5 * (b - a) * gq_weights

    # Evaluate
    return np.sum(weights * f(nodes))


def get_kde_params(obs_prior: np.ndarray, obs, obs_err):
    ## Calculates and stores a bunch of parameters related to a kde distribution
    ## obs_err = np.inf signals that we're using the prior.

    # Get ensemble size so we don't have to keep using .shape[0]
    nens = obs_prior.shape[0]
    params = {'ens' : obs_prior, 'nens' : nens, 'i_nens' : 1.0 / nens, 'obs' : obs, 'obs_err' : obs_err}

    # Get kernel bandwidths
    params["bandwidths"]   = get_kde_bandwidths(obs_prior)
    params["i_bandwidths"] = 1.0 / params["bandwidths"]

    if (obs_err == np.inf):
        params["is_prior"] = True
        params["normalization_constant"] = 1.0
    else:
        ## Posterior distribution, so we need to use quadrature, so we
        ## pre-calculate the cdf at "edges"
        params["is_prior"] = False

        # Get edges of the subintervals on which the pdf is smooth
        edges = np.sort(np.concatenate((obs_prior - params["bandwidths"],
                                        obs_prior + params["bandwidths"])))
        params["edges"] = edges

        # get cdf values evaluated at edges
        cdf_at_edges = np.zeros_like(edges)
        params["normalization_constant"] = 1.0  # placeholder before we compute the actual value
        post_pdf = lambda x: kde_pdf(x, params)
        for i in range(1,2*nens):
            cdf_at_edges[i] = cdf_at_edges[i-1] + gauss_quad(edges[i-1], edges[i], post_pdf)
        params["normalization_constant"] = 1.0 / np.array(cdf_at_edges[-1])  # Use np.array to force a copy
        cdf_at_edges[:] *= params["normalization_constant"]
        params["cdf_at_edges"] = cdf_at_edges
    return params


def kde_pdf(x, params):
    ## Evaluates the kde approximation to the pdf at x. params is a dict set above.
    kde_pdf = 0.0  # Initialize
    if (params["is_prior"]):
        for i in range(params['nens']):  # This is a reduction loop
            kde_pdf += params["i_bandwidths"][i] * epanechnikov_kernel( (x - params["ens"][i]) * params["i_bandwidths"][i] )
        kde_pdf *= params["i_nens"] * params["normalization_constant"]
    else:
        for i in range(params['nens']):  # This is a reduction loop
            kde_pdf += params["i_bandwidths"][i] * epanechnikov_kernel( (x - params["ens"][i]) * params["i_bandwidths"][i] )
        kde_pdf *= params["i_nens"] * params["normalization_constant"] \
                 * np.exp(-0.5 * ((x - params["obs"]) / params["obs_err"])**2)  # TODO: Enable non-Gaussian likelihoods
    return kde_pdf


def kde_cdf(x, params):
    ## Evaluates the cdf at x.
    ## Whether it's prior or posterior is defined by the param dict that is passed in.
    if (params["is_prior"]):
        return np.sum( epanechnikov_cdf( (x - params["ens"]) * params["i_bandwidths"] ) ) * params["i_nens"]
    bin_index = np.digitize(x, params["edges"])
    if (bin_index == 0):
        return 0.0
    if (bin_index == params["edges"].shape[0]):
        return 1.0
    post_pdf = lambda t : kde_pdf(t, params)
    return params["cdf_at_edges"][bin_index-1] + gauss_quad(params["edges"][bin_index-1], x, post_pdf)

