import numpy as np
import unittest
from assim_tools.analysis import *

class TestAnalysis(unittest.TestCase):

    def test_ensemble_transform_weights(self):
        ##test assimilation of 1 obs
        obs = np.array([1.])
        obs_err = np.array([0.1])
        obs_prior = np.array([[1.], [0.5]])
        local_factor = np.array([1.])
        weights = ensemble_transform_weights(obs, obs_err, obs_prior, 'ETKF', local_factor)
        self.assertAlmostEqual(weights[0,0], 1.09904573)
        self.assertAlmostEqual(weights[1,0], -0.09904573)
        self.assertAlmostEqual(weights[0,1], 0.8268802)
        self.assertAlmostEqual(weights[1,1], 0.1731198)


if __name__ == '__main__':
    unittest.main()

