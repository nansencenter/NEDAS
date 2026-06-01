import numpy as np

def comp_dt(x, F):
    """
    Compute Lorenz-96 model tendency dx/dt.
    """
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def adv_1step(x, F, delta_t, mean_velocity, 
              pert_velocity_multiplier, diffusion_coef, e_folding, sink_rate,
              bound_above_is_one=True, positive_tracer=True):
    """
    Perform one advection step + RK4 for Lorenz-96 + diffusion + sources/sinks.
    (adapted from DART/models/lorenz_96_tracer_advection)
    
    Parameters:
        x : numpy array, shape (model_size,)
            Current state: positions, tracer, and sources.
        delta_t : float
            Time step
        mean_velocity : float
            Base velocity for tracer advection
        pert_velocity_multiplier : float
            Multiplier for perturbing velocity
        diffusion_coef : float
            Diffusion coefficient
        e_folding : float
            Exponential sink rate
        sink_rate : float
            Additional uniform sink rate
        bound_above_is_one : bool
            If True, adjust tracer values for upper bound
        positive_tracer : bool
            If True, tracer cannot go below zero
    """
    model_size = len(x)
    grid_size = model_size // 3
    x_new = x.copy()
    q_new = x[grid_size:2*grid_size].copy()  # QTY_TRACER_CONCENTRATION
    if bound_above_is_one:
        q_new -= 1.0
    source = x[2*grid_size:model_size]

    # Upstream semi-Lagrangian advection
    q_tmp = np.zeros(grid_size)
    indices = np.arange(1, grid_size + 1)
    # Compute velocities and target locations
    velocity = (mean_velocity + x[0:grid_size]) * pert_velocity_multiplier
    if np.any(np.abs(velocity * delta_t) > grid_size):
        raise ValueError("Lagrangian Velocity ridiculously large")
    target_loc = indices - velocity * delta_t
    low = np.floor(target_loc).astype(int)
    hi = low + 1
    frac = target_loc - low
    # Apply periodic boundary conditions using modulo
    low = (low - 1) % grid_size
    hi = (hi - 1) % grid_size
    q_tmp = (1 - frac) * q_new[low] + frac * q_new[hi]
    q_new = q_tmp

    # Diffusion
    q_diff = diffusion_coef * delta_t * (np.roll(q_new, 1) + np.roll(q_new, -1) - 2 * q_new)
    q_new += q_diff * delta_t

    # Add source and sink
    q_new += source * delta_t
    q_new *= np.exp(-e_folding * delta_t)
    # Additional uniform sink
    if positive_tracer:
        q_new = np.maximum(0.0, q_new - sink_rate * delta_t)
    else:
        q_new = np.minimum(0.0, q_new + sink_rate * delta_t)

    # Add back 1 for tracer if needed
    if bound_above_is_one:
        q_new += 1.0
    x_new[grid_size:2*grid_size] = q_new

    # RK4 for Lorenz-96 states
    x0 = x[0:grid_size].copy()
    x1 = delta_t * comp_dt(x0, F)
    inter = x[0:grid_size] + x1 / 2
    x2 = delta_t * comp_dt(inter, F)
    inter = x[0:grid_size] + x2 / 2
    x3 = delta_t * comp_dt(inter, F)
    inter = x[0:grid_size] + x3
    x4 = delta_t * comp_dt(inter, F)
    x_new[0:grid_size] = x0 + x1/6 + x2/3 + x3/3 + x4/6

    return x_new

def M_nl(x, T, F, dt, mean_velocity, pert_velocity_multiplier, diffusion_coef, e_folding, sink_rate, bound_above_is_one=True, positive_tracer=True):
    """
    Nonlinear model propagator for time duration T.
    """
    for _ in range(int(T/dt)):
        x = adv_1step(x, F, dt, mean_velocity, pert_velocity_multiplier, diffusion_coef, e_folding, sink_rate, bound_above_is_one, positive_tracer)
    return x
