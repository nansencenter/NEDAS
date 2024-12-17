import numpy as np
from utils.njit import njit

@njit(cache=True)
def adjust_dp(dp, depth, onem):
    kdm, jdm, idm = dp.shape
    dp_ = dp.copy()
    press = np.zeros(kdm + 1)

    for j in range(jdm):
        for i in range(idm):
            # Move negative layers to neighbouring layers.
            for k in range(kdm - 1):
                dp[k+1, j, i] += min(0.0, dp[k, j, i])
                dp[k, j, i] = max(dp[k, j, i], 0.0)

            # Go backwards to fix lowermost layer.
            for k in range(kdm - 1, 2, -1):
                dp[k, j, i-1] += min(0.0, dp[k, j, i])
                dp[k, j, i] = max(dp[k, j, i], 0.0)

            # No layers below the sea bed.
            press[0] = 0.0
            for k in range(kdm - 1):
                press[k+1] = press[k] + dp[k, j, i]
                press[k+1] = min(depth[j, i] * onem, press[k+1])
            press[kdm] = depth[j, i] * onem

            for k in range(kdm):
                dp[k, j, i] = press[k+1] - press[k]

            if depth[j, i] > 100000.0 or depth[j, i] < 1.0:
                dp[:, j, i] = dp_[:, j, i]

    return dp
