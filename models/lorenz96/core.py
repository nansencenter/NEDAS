import numpy as np

##nonlinear forward model
def M_nl(x, F, T, dt):
    """
    Lorenz 1996 model with 40 variables, nonlinear advance_time function
    Input:
    -x: np.array, the model state
    -F: parameter, default is 8
    -T: duration of the simulation
    -dt: model time step
    Output:
    -x: np.array, the updated model state after simulation
    """
    for n in range(int(T/dt)):
        x += ((np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F) * dt
    return x

