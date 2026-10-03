"""
Tests for the small models from DART and DAPPER (NEDAS/models/ode_model.py and its subclasses).

Each model was checked once against its source, one time step from the same state, to round-off:
lorenz63, lorenz84, lorenz05, ikeda, lotka_volterra and ks against DAPPER (dapper/mods), and
nine_var and lorenz96_2scale against DART's compiled model_mod (adv_1step). Those need the other
packages; here the models are checked against reductions and invariants of their equations, and
for the NEDAS interface (io in both modes, run, truth and ensemble generation).
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
import numpy as np
from NEDAS.core import Context
from NEDAS.models import get_model_class

SMALL_MODELS = ['lorenz63', 'lorenz84', 'lorenz05', 'lorenz96_2scale', 'nine_var', 'ikeda', 'lotka_volterra', 'ks']


class TestInterface(unittest.TestCase):
    def setUp(self):
        self.c = Context()
        self.dir = tempfile.mkdtemp()
        self.time = datetime(2001, 1, 1)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_grid_variables_and_initial_condition(self):
        for name in SMALL_MODELS:
            with self.subTest(model=name):
                m = get_model_class(name)(context=self.c)
                x = m.initial_condition(0)
                self.assertEqual(x.shape, (m.state_size,))
                self.assertTrue(np.all(np.isfinite(x)))
                for v in m.variables:
                    self.assertEqual(m.get_field(x, v).shape, m.grid.x.shape)
                # different seeds give different states, the same seed the same one
                self.assertFalse(np.allclose(m.initial_condition(1), x))
                np.testing.assert_array_equal(m.initial_condition(0), x)

    def test_io_round_trip_in_both_modes(self):
        for name in SMALL_MODELS:
            for io_mode in ('online', 'offline'):
                with self.subTest(model=name, io_mode=io_mode):
                    m = get_model_class(name)(context=self.c, io_mode=io_mode)
                    kw = dict(path=os.path.join(self.dir, name), time=self.time, member=3, tag='current')
                    x = m.initial_condition(0)
                    m.set_state(x, **kw)
                    for v in m.variables:
                        fld = m.read_var(**kw, name=v)
                        # writing back an unchanged field leaves the state alone
                        m.write_var(fld, **kw, name=v)
                        np.testing.assert_allclose(m.get_state(**kw), x, rtol=0, atol=1e-12)
                        # a change is read back
                        m.write_var(fld + 0.5, **kw, name=v)
                        np.testing.assert_allclose(m.read_var(**kw, name=v), fld + 0.5, rtol=0, atol=1e-12)
                        m.set_state(x, **kw)

    def test_run_advances_by_the_forecast_period(self):
        for name in SMALL_MODELS:
            for io_mode in ('online', 'offline'):
                with self.subTest(model=name, io_mode=io_mode):
                    m = get_model_class(name)(context=self.c, io_mode=io_mode)
                    kw = dict(path=os.path.join(self.dir, name), time=self.time, member=0, tag='current', forecast_period=12)
                    x = m.initial_condition(0)
                    m.set_state(x, **kw)
                    m.run(**kw)
                    later = m.get_state(**{**kw, 'time': self.time + timedelta(hours=12)})
                    np.testing.assert_array_equal(later, m.advance(x, m.nsteps(12)))
                    self.assertFalse(np.allclose(later, x))

    def test_forecast_period_must_be_a_multiple_of_the_time_step(self):
        m = get_model_class('lorenz63')(context=self.c)  # 0.24 h time step
        with self.assertRaises(AssertionError):
            m.nsteps(1.)

    def test_preprocess_keeps_the_prior(self):
        m = get_model_class('lorenz63')(context=self.c, io_mode='online')
        kw = dict(time=self.time, member=0, tag='current')
        x = m.initial_condition(0)
        m.set_state(x, **kw)
        m.preprocess(**kw)
        m.write_var(m.read_var(**kw, name='state') + 1., **kw, name='state')
        np.testing.assert_array_equal(m.get_state(**{**kw, 'tag': 'prior'}), x)
        m.postprocess(**kw)
        np.testing.assert_array_equal(m.get_state(**{**kw, 'tag': 'post'}), x + 1.)

    def test_truth_and_ensemble(self):
        for io_mode in ('online', 'offline'):
            with self.subTest(io_mode=io_mode):
                c = Context()
                c.config.time_start = self.time
                c.config.time_end = self.time + timedelta(hours=24)
                d = os.path.join(self.dir, io_mode)
                m = get_model_class('lorenz84')(context=c, io_mode=io_mode,
                                                truth_dir=os.path.join(d, 'truth'), ens_init_dir=os.path.join(d, 'ens'))
                m.generate_truth(tag='truth', forecast_period=6)
                times = [self.time + timedelta(hours=h) for h in range(0, 25, 6)]
                truth = [m.get_state(path=m.truth_dir, time=t, member=None, tag='truth') for t in times]
                for a, b in zip(truth[:-1], truth[1:]):
                    np.testing.assert_allclose(b, m.advance(a, m.nsteps(6)))
                # climatological members differ from the truth and each other
                for mem in range(2):
                    m.generate_init_ensemble(path=m.ens_init_dir, tag='current', member=mem)
                ens = [m.get_state(path=m.ens_init_dir, time=self.time, member=mem, tag='current') for mem in range(2)]
                self.assertFalse(np.allclose(ens[0], ens[1]))
                self.assertFalse(np.allclose(ens[0], truth[0]))
                # or the truth plus noise
                m.init_ens_mode, m.init_ens_sd = 'truth', 1e-3
                m.generate_init_ensemble(path=m.ens_init_dir, tag='current', member=5)
                x = m.get_state(path=m.ens_init_dir, time=self.time, member=5, tag='current')
                self.assertLess(np.abs(x - truth[0]).max(), 0.01)


class TestDynamics(unittest.TestCase):
    def setUp(self):
        self.c = Context()
        self.rng = np.random.default_rng(0)

    def l96_dxdt(self, x, F):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    def test_lorenz63_tendency(self):
        m = get_model_class('lorenz63')(context=self.c)
        np.testing.assert_allclose(m.dxdt(np.array([1., 2., 3.])), [10., 28 - 2 - 3, 2 - 8.])

    def test_lorenz63_stays_on_the_attractor(self):
        m = get_model_class('lorenz63')(context=self.c)
        x = m.advance(m.initial_condition(0), 5000)
        self.assertLess(np.abs(x[:2]).max(), 30.)
        self.assertTrue(0. < x[2] < 60.)

    def test_lorenz05_model_I_is_lorenz96(self):
        # K = 1 and I = 1 reduce Lorenz (2005) Model III to Lorenz-96
        m = get_model_class('lorenz05')(context=self.c, nx=40, K=1, I=1, F=8.)
        x = self.rng.normal(0, 3, 40)
        np.testing.assert_allclose(m.dxdt(x), self.l96_dxdt(x, 8.), atol=1e-12)

    def test_lorenz05_decomposition(self):
        m = get_model_class('lorenz05')(context=self.c)
        z = m.initial_condition(0)
        x, y = m.decompose(z)
        np.testing.assert_allclose(x + y, z)
        # the filter keeps a long wave and removes the grid-scale one
        i = np.arange(m.nx)
        long_wave = np.sin(2 * np.pi * 3 * i / m.nx)
        np.testing.assert_allclose(m.decompose(long_wave)[0], long_wave, atol=1e-3)
        self.assertLess(np.abs(m.decompose((-1.) ** i)[0]).max(), 1e-3)

    def test_lorenz96_2scale_uncoupled_slow_variables_are_lorenz96(self):
        m = get_model_class('lorenz96_2scale')(context=self.c, coupling_h=0.)
        x = self.rng.normal(0, 3, m.state_size)
        np.testing.assert_allclose(m.dxdt(x)[:m.K], self.l96_dxdt(x[:m.K], m.F))

    def test_lorenz96_2scale_slow_field_on_the_fast_grid(self):
        m = get_model_class('lorenz96_2scale')(context=self.c)
        x = m.initial_condition(0)
        X = m.get_field(x, 'slow')
        np.testing.assert_array_equal(X.reshape(m.K, m.J), np.repeat(x[:m.K, None], m.J, axis=1))
        # a change of the field moves X_k by its mean over the J points, Y is not touched
        inc = self.rng.normal(0, 1, X.shape)
        y = x.copy()
        m.set_field(y, 'slow', X + inc)
        np.testing.assert_allclose(y[:m.K], x[:m.K] + inc.reshape(m.K, m.J).mean(axis=1))
        np.testing.assert_array_equal(y[m.K:], x[m.K:])

    def test_nine_var_matches_the_dart_loop(self):
        # a literal transcription of comp_dt in DART models/9var/model_mod.f90
        m = get_model_class('nine_var')(context=self.c)
        a, b, f, h = [1., 1., 3.], [-1.5, -1.5, 0.5], [0.1, 0., 0.], [-1., 0., 0.]
        nu = kappa = 1. / 48.
        c, g = 0.8660254, m.g
        s = self.rng.normal(0, 0.1, 9)
        x, y, z = s[0:3], s[3:6], s[6:9]
        dx, dy, dz = np.zeros(3), np.zeros(3), np.zeros(3)
        for i1 in range(1, 4):
            j1, k1 = i1 % 3 + 1, (i1 + 1) % 3 + 1
            i, j, k = i1 - 1, j1 - 1, k1 - 1
            dx[i] = (a[i]*b[i]*x[j]*x[k] - c*(a[i] - a[k])*x[j]*y[k] + c*(a[i] - a[j])*y[j]*x[k]
                     - 2*c**2*y[j]*y[k] - nu*a[i]**2*x[i] + a[i]*y[i] - a[i]*z[i]) / a[i]
            dy[i] = (-1.*a[k]*b[k]*x[j]*y[k] - a[j]*b[j]*y[j]*x[k] + c*(a[k] - a[j])*y[j]*y[k]
                     - a[i]*x[i] - nu*a[i]**2*y[i]) / a[i]
            dz[i] = (-1.*b[k]*x[j]*(z[k] - h[k]) - b[j]*(z[j] - h[j])*x[k] + c*y[j]*(z[k] - h[k])
                     - c*(z[j] - h[j])*y[k] + g*a[i]*x[i] - kappa*a[i]*z[i] + f[i])
        np.testing.assert_allclose(m.dxdt(s), np.concatenate([dx, dy, dz]), atol=1e-14)

    def test_ikeda_map(self):
        m = get_model_class('ikeda')(context=self.c)
        # from the origin: t = 0.4 - 6, and the map gives (1, 0)
        np.testing.assert_allclose(m.step(np.zeros(2)), [1., 0.])
        x = m.advance(m.initial_condition(0), 1000)
        self.assertLess(np.abs(x).max(), 1. / (1. - m.u) + 1.)  # |x_n| is bounded by 1 + u |x_{n-1}|

    def test_lotka_volterra_stays_positive(self):
        m = get_model_class('lotka_volterra')(context=self.c)
        x = m.initial_condition(0)
        for _ in range(20):
            x = m.advance(x, 100)
            self.assertTrue(np.all(x > 0.) and np.all(x < 1.5))

    def test_ks_conserves_the_mean_and_is_chaotic(self):
        m = get_model_class('ks')(context=self.c)
        x = m.initial_condition(0)
        y = m.advance(x, 400)
        self.assertAlmostEqual(y.mean(), x.mean(), places=10)
        self.assertLess(np.abs(y).max(), 10.)
        # nearby states separate
        d0 = 1e-6
        x2 = x + d0 * self.rng.normal(0, 1, x.shape)
        self.assertGreater(np.abs(m.advance(x2, 400) - y).max(), 1e3 * d0)


if __name__ == '__main__':
    unittest.main()
