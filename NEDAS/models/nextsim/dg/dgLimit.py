import numpy as np

# Note: only DGVector<6> PSIImpl<6,3> implemented for now

# Define PSI matrix based on values from PSIImpl<6,3> in
# nextsimdg/dynamics/src/include/codeGenerationDGinGauss.hpp
PSI = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [-0.3872983346207417, 0.0, 0.3872983346207417, -0.3872983346207417, 0.0,
     0.3872983346207417, -0.3872983346207417, 0.0, 0.3872983346207417],
    [-0.3872983346207417, -0.3872983346207417, -0.3872983346207417, 0.0, 0.0, 0.0,
     0.3872983346207417, 0.3872983346207417, 0.3872983346207417],
    [0.0666666666666667, -0.08333333333333333, 0.0666666666666667, 0.0666666666666667,
     -0.08333333333333333, 0.0666666666666667, 0.0666666666666667,
     -0.08333333333333333, 0.0666666666666667],
    [0.0666666666666667, 0.0666666666666667, 0.0666666666666667, 0.0666666666666667,
     -0.08333333333333333, -0.08333333333333333, -0.08333333333333333,
     0.0666666666666667, 0.0666666666666667],
    [0.15000000000000002, -0.0, -0.15000000000000002, -0.0, 0.0, 0.0,
     -0.15000000000000002, 0.0, 0.15000000000000002]
])

def limit_min(dg_field, min_value):
    dg_field_limited = dg_field.copy()

    ##step 1: clamp the first coefficient (mean value)
    dg_field_limited[..., 0] = np.maximum(dg_field_limited[..., 0], min_value)

    # step 2: compute values at Gauss nodes
    gauss_values = np.einsum("...j,jk->...k", dg_field_limited, PSI)
    min_gauss = np.min(gauss_values, axis=-1)

    # step 3: determine necessary scaling for higher-order coefficients
    ind = np.where(min_gauss < min_value)
    scaling = (dg_field_limited[..., 0][ind] - min_value) / (dg_field_limited[..., 0][ind] - min_gauss[ind])

    # apply scaling
    for i in range(1, 6):
        dg_field_limited[..., i][ind] *= scaling

    return dg_field_limited


def limit_max(dg_field, max_value):
    dg_field_limited = dg_field.copy()

    # step 1: clamp the first coefficient (mean value)
    dg_field_limited[..., 0] = np.minimum(dg_field_limited[..., 0], max_value)

    # step 2: compute values at Gauss nodes
    gauss_values = np.einsum("...j,jk->...k", dg_field_limited, PSI)
    max_gauss = np.max(gauss_values, axis=-1)

    # step 3: determine necessary scaling for higher-order coefficients
    ind = np.where(max_gauss > max_value)
    scaling = (max_value - dg_field_limited[..., 0][ind]) / (max_gauss[ind] - dg_field_limited[..., 0][ind])

    # apply scaling
    for i in range(1, 6):
        dg_field_limited[..., i][ind] *= scaling

    return dg_field_limited
