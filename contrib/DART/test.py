import numpy as np
from probit import dart_wrapper

ens_size = 20
state_ens_in = np.random.rand(ens_size).astype(np.float64)
distribution_type = 2  # for example, LOG_NORMAL_DISTRIBUTION
use_input_p = True
bounded_below = False
bounded_above = False
lower_bound = 0.0
upper_bound = 1.0

probit_ens = np.zeros_like(state_ens_in)
ierr = np.array(0, dtype=np.int32)

probit_ens, ierr = dart_wrapper.py_transform_to_probit(
                ens_size, state_ens_in, distribution_type,
                use_input_p, bounded_below, bounded_above, lower_bound, upper_bound
                )

print("Output:", probit_ens)
print("Error code:", ierr)

