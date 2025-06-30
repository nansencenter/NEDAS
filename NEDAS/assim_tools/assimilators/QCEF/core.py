import numpy as np
from NEDAS.utils.njit import njit
from NEDAS.assim_tools.assimilators.serial import SerialAssimilator
from scipy.optimize import root_scalar

class QCEFAssimilator(SerialAssimilator):
    def obs_increment(self, obs_prior, obs, obs_err):
        return obs_increment_qcef(obs_prior, obs, obs_err)

    def update_local_state(self, state_prior, obs_prior, obs_incr,
                        state_h_dist, state_v_dist, state_t_dist,
                        hroi, vroi, troi,
                        h_local_func, v_local_func, t_local_func) -> None:
        return update_local_state_linear(state_prior, obs_prior, obs_incr,
                                         state_h_dist, state_v_dist, state_t_dist,
                                         hroi, vroi, troi,
                                         h_local_func, v_local_func, t_local_func)

    def update_local_obs(self, obs_data, used, obs_prior, obs_incr,
                         h_dist, v_dist, t_dist,
                         hroi, vroi, troi,
                         h_local_func, v_local_func, t_local_func) -> None:
        return update_local_obs_linear(obs_data, used, obs_prior, obs_incr,
                                       h_dist, v_dist, t_dist,
                                       hroi, vroi, troi,
                                       h_local_func, v_local_func, t_local_func)

    def transform_ens_state_forward(self, state_data):
        #here the implementation of probit transform_to_probit for all state variables
        pass

    def transform_ens_state_backward(self, state_data):
        pass

    def transform_ens_obs_forward(self, obs_data):
        #here the implementation of probit transform_to_probit for all obs variables
        pass

    def transform_ens_obs_backward(self, obs_data):
        pass

def transform_to_probit():
    pass

def transform_from_probit():
    pass

def obs_increment_qcef(obs_prior, obs, obs_err) -> np.ndarray:

    obs_increment = np.zeros_like(obs_prior)

    # If all members are equal, return zero increments.
    d_max = np.absolute(obs_prior[0] - obs_prior[:]).max()
    if (d_max <= 0.0):
        return obs_increment

    # Get prior and posterior distribution parameters
    params_prior = get_kde_params(obs_prior, 0.0, np.Inf)
    params_post  = get_kde_params(obs_prior, obs, obs_err)

    for i in range(params["nens"]):  # Loop iterations are independent, parallelizable
        u = kde_cdf(obs_prior[i], params_prior)
        f = lambda x: kde_cdf(x, params_post) - u
        fprime = lambda x: kdf_pdf(x, params_post)
        obs_increment[i] = root_scalar(f, fprime=fprime, x0=obs_prior[i]).root - obs_prior[i]  # TODO: Add logic to catch errors. Maybe a better initial guess.

    return obs_increment

@njit
def update_local_state_linear(state_data, obs_prior, obs_incr,
                              h_dist, v_dist, t_dist,
                              hroi, vroi, troi,
                              h_local_func, v_local_func, t_local_func) -> None:

    nens, nfld, nloc = state_data.shape

    h_lfactor = h_local_func(h_dist, hroi)
    v_lfactor = v_local_func(v_dist, vroi)
    t_lfactor = t_local_func(t_dist, troi)

    nloc_sub = np.where(h_lfactor>0)[0]  ##subset of range(nloc) to update

    ##TODO: impact_on_state missing
    lfactor = np.zeros((nfld, nloc))
    for l in nloc_sub:
        for n in range(nfld):
            lfactor[n, l] = h_lfactor[l] * v_lfactor[n, l] * t_lfactor[n]

    state_data[:, :, nloc_sub] = update_ensemble(state_data[:, :, nloc_sub], obs_prior, obs_incr, lfactor[:, nloc_sub])

@njit
def update_local_obs_linear(obs_data, used, obs_prior, obs_incr,
                            h_dist, v_dist, t_dist,
                            hroi, vroi, troi,
                            h_local_func, v_local_func, t_local_func):

    ##distance between local obs_data and the obs being assimilated
    h_lfactor = h_local_func(h_dist, hroi)
    v_lfactor = v_local_func(v_dist, vroi)
    t_lfactor = t_local_func(t_dist, troi)

    lfactor = h_lfactor * v_lfactor * t_lfactor

    ##update the unused obs within roi
    ind = np.where(np.logical_and(~used, lfactor>0))[0]

    obs_data[:, ind] = update_ensemble(obs_data[:, ind], obs_prior, obs_incr, lfactor[ind])

@njit
def update_ensemble(ens_prior, obs_prior, obs_incr, local_factor) -> np.ndarray:
    nens = ens_prior.shape[0]
    ens_post = ens_prior.copy()

    ##obs-space statistics
    obs_prior_mean = np.mean(obs_prior)
    obs_prior_var = np.sum((obs_prior - obs_prior_mean)**2) / (nens-1)

    ##if there is no prior spread, don't update at all
    if obs_prior_var == 0:
        return ens_post

    cov = np.zeros(ens_prior.shape[1:])
    for m in range(nens):
        cov += ens_prior[m, ...] * (obs_prior[m] - obs_prior_mean) / (nens-1)

    reg_factor = cov / obs_prior_var

    ##the updated posterior ensemble
    for m in range(nens):
        ens_post[m, ...] = ens_prior[m, ...] + local_factor * reg_factor * obs_incr[m]

    return ens_post

@njit
def epanechnikov_kernel(x) -> np.ndarray:
    return 0.75 * max(0.0, 1.0 - x**2)

@njit
def epanechnikov_cdf(x) -> np.ndarray:
    x_truncated = min(1.0, max(-1.0, x))
    return 0.25 * (2.0 + 3.0 * x_truncated - x_truncated**3)

@njit
def get_kde_bandwidths(obs_prior) -> np.ndarray:
    ## uses the observation-space forecast ensemble to compute kernel bandwidths
    nens = obs_prior.shape[0]
    d_max = np.absolute(obs_prior[0] - obs_prior[:]).max()
    #if (d_max <= 0.0): There should be a warning/error here. Can't continue if all ensemble members have the same value.
    ens_mean = np.mean(obs_prior)
    ens_sd   = np.std(obs_prior)
    h0 = 2.0 * ens_sd / (ens_size**0.2)  # This would be the kernel width if the widths were not adaptive.
                                         # It would be better to use min(sd, iqr/1.34) but don't want to compute iqr
    k = np.floor( np.sqrt(nens) )  # distance to kth nearest neighbor is used to set bandwidth for k defined here
    f_tilde = np.zeros(nens)
    for i in range(nens):
        ## This loop can fail if the kth nearest neighbor has distance 0, in which case you need to
        ## search for the first neighbor that has a nonzero distance.
        dist  = np.absolute(obs_prior[0] - obs_prior[:]).sort()
        d_max = dist.max()
        dist  = np.where(dist <= 1.E-3 * d_max, 0.0, dist)  # replace small distances with 0
        f_tilde[i] = 0.5 * (k - 1) / (nens * dist[k])       # Initial density estimate
    f_tilde[:] = f_tilde[:] / np.max(f_tilde[:])            # Avoids overflow in the next line
    g = f_tilde.prod()**(1.0 / nens)
    lamda = np.sqrt(g / f_tilde)
    return h0 * lamda

@njit
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

@njit
def get_kde_params(obs_prior, obs, obs_err):
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
        params["edges"] = np.sort(edges)

        # get cdf values evaluated at edges
        cdf_at_edges = np.zeros(2*nens)
        params["normalization_constant"] = 1.0  # placeholder before we compute the actual value
        post_pdf = lambda x: kde_pdf(x, params)
        for i in range(1,2*nens):
            cdf_at_edges[i] = cdf_at_edges[i-1] + gauss_quad(edges[i-1], edges[i], post_pdf)
        params["normalization_constant"] = 1.0 / np.array(cdf_at_edges[-1])  # Use np.array to force a copy
        cdf_at_edges[:] *= params["normalization_constant"]
        params["cdf_at_edges"] = cdf_at_edges
    return params

@njit
def kde_pdf(x, params):
    ## Evaluates the kde approximation to the pdf at x. params is a dict set above.
    kde_pdf = 0.0  # Initialize
    if (params["is_prior"]):
        for i in range(nens):  # This is a reduction loop
            kde_pdf += params["i_bandwidths"] * epanechnikov_kernel( (x - params["ens"][i]) * params["i_bandwidths"][i] )
        kde_pdf *= params["i_nens"] * params["normalization_constant"]
    else:
        for i in range(nens):  # This is a reduction loop
            kde_pdf += params["i_bandwidths"] * epanechnikov_kernel( (x - params["ens"][i]) * params["i_bandwidths"][i] )
        kde_pdf *= params["i_nens"] * params["normalization_constant"] \
                 * np.exp(-0.5 * ((x - params["obs"]) / params["obs_err"])**2)  # TODO: Enable non-Gaussian likelihoods
    return kde_pdf

@njit
def kde_cdf(x, params):
    ## Evaluates the cdf at x.
    ## Whether it's prior or posterior is defined by the param dict that is passed in.
    if (params["is_prior"]):
        return np.sum( epanechnikov_cdf( (x - params["obs_prior"]) * params["i_bandwidths"] ) ) * params["i_nens"]
    bin_index = np.digitize(x, params["edges"])
    if (bin_index == 0):
        return 0.0
    if (bin_index == params["edges"].shape[0]):
        return 1.0
    post_pdf = lambda t : kde_pdf(t, params)
    return params["cdf_at_edges"][bin_index-1] + gauss_quad(params["edges"][bin_index-1], x, post_pdf)