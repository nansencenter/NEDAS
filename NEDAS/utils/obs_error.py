"""
Observation error models for synthetic observations.

Three are supported, selected by ``err.type`` in the observation definition:

``normal``
    obs = truth + std * N(0, 1). ``std`` is in the variable's units and is the same for every
    observation.

``lognormal``
    obs = max(truth, floor) * exp(std * N(0, 1)). ``std`` is the standard deviation of the log of
    the error, so it is dimensionless (0.3 is roughly a 30% relative error), and the observation
    is always positive. For a quantity bounded below by zero (humidity, a concentration) this
    keeps the noise on the right side of the bound, which additive noise cannot. The error is
    multiplicative, so the assimilator's error standard deviation differs per observation; it is
    std * obs, the first-order value, and is what the assimilator is given.

    This first-order std depends on the noise draw itself (a low draw gets a small error and
    more weight), which biases the analysis low; ``truncated_normal`` is the error model to use
    for bounded quantities when the filters are to be compared.

    ``floor`` is the smallest value the truth is taken to have. It is needed because a
    multiplicative error on exactly zero is zero, which is both a degenerate observation (it
    carries an error standard deviation of zero) and not strictly positive. Leave it at 0 only
    where the truth is known to be positive.

``truncated_normal``
    obs = truth + std * N(0, 1), redrawn until obs >= ``lower_bound`` (default 0): a normal error
    truncated at the bound, with one ``std`` for every observation, which is also what the
    assimilator is given. This is the error model of the QCEFF studies for bounded quantities
    (Anderson et al. 2024, MWR 152, Part III: truncated normal, constant variance, bound 0); unlike
    the lognormal, an observation can sit at the bound itself.
"""
import numpy as np

KNOWN_TYPES = ('normal', 'lognormal', 'truncated_normal')


def _check(err) -> None:
    if err.type not in KNOWN_TYPES:
        raise ValueError(f"unsupported observation error type '{err.type}', "
                         f"choose one of {', '.join(KNOWN_TYPES)}")
    if err.type == 'lognormal' and not err.floor >= 0:
        raise ValueError(f"lognormal error 'floor' must be >= 0, got {err.floor}")


def perturb_obs(truth: np.ndarray, err) -> np.ndarray:
    """The truth-evaluated observation values with this error model's noise added."""
    _check(err)
    noise = np.random.normal(0, 1, truth.shape)
    if err.type == 'normal':
        return truth + noise * err.std
    if err.type == 'truncated_normal':
        # a truth already below the bound is taken at the bound, so each redraw keeps >= 1/2 chance
        truth = np.maximum(truth, err.lower_bound)
        obs = truth + noise * err.std
        redraw = obs < err.lower_bound
        while redraw.any():
            obs[redraw] = truth[redraw] + np.random.normal(0, 1, redraw.sum()) * err.std
            redraw = obs < err.lower_bound
        return obs
    return np.maximum(truth, err.floor) * np.exp(noise * err.std)


def assimilation_std(obs: np.ndarray, err) -> np.ndarray:
    """The error standard deviation per observation that the assimilator's R is built from."""
    _check(err)
    if err.type in ('normal', 'truncated_normal'):
        # one per location: a vector obs is (2, nobs), its err_std (nobs,)
        return np.full(obs.shape[-1:], err.std * err.infl)
    if obs.ndim > 1:
        raise ValueError(f"{err.type} error on a vector obs: no single std per location")
    return err.std * np.abs(obs) * err.infl
