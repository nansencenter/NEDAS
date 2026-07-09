import numpy as np
import unittest
from datetime import datetime, timezone
from NEDAS.core import Context
from NEDAS.core.types import FieldRecord, ObsRecord, ErrorModel
from NEDAS.assim_tools.transforms import get_transform_funcs
from NEDAS.assim_tools.transforms.identity import Identity


def _field_record() -> FieldRecord:
    return FieldRecord(
        name='test', model_src='test', dtype='float', is_vector=False,
        units='*', err_type='gaussian',
        time=datetime(2000, 1, 1, tzinfo=timezone.utc),
        dt=1.0, k=0, pos=0,
    )


def _obs_record() -> ObsRecord:
    err = ErrorModel(type='gaussian', std=1.0, hcorr=0.0, vcorr=0.0, tcorr=0.0, cross_corr=())
    return ObsRecord(
        name='test', dataset_src='test', model_src='test',
        nobs=0, obs_window_min=-1, obs_window_max=1,
        dtype='float', is_vector=False, units='*', z_units='*',
        time=datetime(2000, 1, 1, tzinfo=timezone.utc),
        dt=1.0, err=err, hroi=100.0, vroi=100.0, troi=1.0, impact_on_variable=(),
    )


class TestGetTransformFuncs(unittest.TestCase):

    def test_explicit_identity_returns_identity(self):
        c = Context()
        c.config.transform_def = {'type': 'identity'}
        funcs = get_transform_funcs(c)
        self.assertIsInstance(funcs[0], Identity)

    def test_list_of_transforms_all_instantiated(self):
        c = Context()
        c.config.transform_def = [{'type': 'identity'}, {'type': 'identity'}]  # type: ignore[assignment]
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
        self.rec = _field_record()
        self.obs_rec = _obs_record()

    def test_forward_state_passthrough(self):
        field = np.random.default_rng(0).normal(0, 1, (50, 50))
        out = self.transform.forward_state(self.c, self.rec, field)
        np.testing.assert_array_equal(out, field)

    def test_backward_state_passthrough(self):
        field = np.random.default_rng(1).normal(0, 1, (50, 50))
        out = self.transform.backward_state(self.c, self.rec, field)
        np.testing.assert_array_equal(out, field)

    def test_forward_obs_passthrough(self):
        obs_seq = {'obs': np.arange(10.0), 'err_std': np.ones(10)}
        out = self.transform.forward_obs(self.c, self.obs_rec, obs_seq)
        self.assertIs(out, obs_seq)

    def test_backward_obs_passthrough(self):
        obs_seq = {'obs': np.arange(10.0)}
        out = self.transform.backward_obs(self.c, self.obs_rec, obs_seq)
        self.assertIs(out, obs_seq)

    def test_forward_backward_roundtrip(self):
        field = np.random.default_rng(2).normal(0, 1, (10, 10))
        roundtrip = self.transform.backward_state(
            self.c, self.rec, self.transform.forward_state(self.c, self.rec, field)
        )
        np.testing.assert_array_equal(roundtrip, field)


if __name__ == '__main__':
    unittest.main()
