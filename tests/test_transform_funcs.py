import numpy as np
import unittest
from NEDAS.core import Context
from NEDAS.assim_tools.transforms import get_transform_funcs
from NEDAS.assim_tools.transforms.identity import Identity


class TestGetTransformFuncs(unittest.TestCase):

    def test_none_transform_def_defaults_to_identity(self):
        c = Context()
        c.config.transform_def = None
        funcs = get_transform_funcs(c)
        self.assertEqual(len(funcs), 1)
        self.assertIsInstance(funcs[0], Identity)

    def test_explicit_identity_returns_identity(self):
        c = Context()
        c.config.transform_def = {'type': 'identity'}
        funcs = get_transform_funcs(c)
        self.assertIsInstance(funcs[0], Identity)

    def test_list_of_transforms_all_instantiated(self):
        c = Context()
        c.config.transform_def = [{'type': 'identity'}, {'type': 'identity'}]
        funcs = get_transform_funcs(c)
        self.assertEqual(len(funcs), 2)

    def test_missing_type_key_raises_key_error(self):
        c = Context()
        c.config.transform_def = {'decompose_obs': False}
        with self.assertRaises(KeyError):
            get_transform_funcs(c)

    def test_unknown_type_raises_not_implemented(self):
        c = Context()
        c.config.transform_def = {'type': 'no_such_transform'}
        with self.assertRaises(NotImplementedError):
            get_transform_funcs(c)


class TestIdentityTransform(unittest.TestCase):

    def setUp(self):
        self.c = Context()
        self.c.config.transform_def = {'type': 'identity'}
        self.transform = get_transform_funcs(self.c)[0]

    def test_forward_state_passthrough(self):
        field = np.random.default_rng(0).normal(0, 1, (50, 50))
        out = self.transform.forward_state(self.c, {}, field)
        np.testing.assert_array_equal(out, field)

    def test_backward_state_passthrough(self):
        field = np.random.default_rng(1).normal(0, 1, (50, 50))
        out = self.transform.backward_state(self.c, {}, field)
        np.testing.assert_array_equal(out, field)

    def test_forward_obs_passthrough(self):
        obs_seq = {'obs': np.arange(10.0), 'err_std': np.ones(10)}
        out = self.transform.forward_obs(self.c, {}, obs_seq)
        self.assertIs(out, obs_seq)

    def test_backward_obs_passthrough(self):
        obs_seq = {'obs': np.arange(10.0)}
        out = self.transform.backward_obs(self.c, {}, obs_seq)
        self.assertIs(out, obs_seq)

    def test_forward_backward_roundtrip(self):
        field = np.random.default_rng(2).normal(0, 1, (10, 10))
        roundtrip = self.transform.backward_state(
            self.c, {}, self.transform.forward_state(self.c, {}, field)
        )
        np.testing.assert_array_equal(roundtrip, field)


if __name__ == '__main__':
    unittest.main()
