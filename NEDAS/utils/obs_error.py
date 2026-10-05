"""
Observation error models for synthetic observations.

Two are supported, selected by ``err.type`` in the observation definition:

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

    ``floor`` is the smallest value the truth is taken to have. It is needed because a
    multiplicative error on exactly zero is zero, which is both a degenerate observation (it
    carries an error standard deviation of zero) and not strictly positive. Leave it at 0 only
    where the truth is known to be positive.
"""
import numpy as np

KNOWN_TYPES = ('normal', 'lognormal')


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
    return np.maximum(truth, err.floor) * np.exp(noise * err.std)


def assimilation_std(obs: np.ndarray, err) -> np.ndarray:
    """The error standard deviation per observation that the assimilator's R is built from."""
    _check(err)
    if err.type == 'normal':
        # one per location: a vector obs is (2, nobs), its err_std (nobs,)
        return np.full(obs.shape[-1:], err.std * err.infl)
    if obs.ndim > 1:
        raise ValueError(f"{err.type} error on a vector obs: no single std per location")
    return err.std * np.abs(obs) * err.infl
