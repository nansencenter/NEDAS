"""
Tests for the DART bgrid_solo (Held-Suarez dry dynamical core) model.

The grid, the wind interpolation and the file layout are checked as they are. The model run
itself needs the nedas_bgrid_advance executable (NEDAS/models/bgrid_solo/build_bgrid_solo.sh,
which needs a DART checkout), without which those tests are skipped.
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
from netCDF4 import Dataset
from NEDAS.core import Context
from NEDAS.models import get_model_class
from NEDAS.models.bgrid_solo.util import vel_to_temp, temp_to_vel, grid_coords, write_restart_file

EXE = os.path.join(os.path.dirname(__import__('NEDAS.models.bgrid_solo', fromlist=['x']).__file__), 'nedas_bgrid_advance')


class TestStaggering(unittest.TestCase):
    def test_grid_coords_match_dart_template(self):
        # DART's perfect_input.nc: temperature points at 3, 9, ... and -87, -81, ..., velocity points
        # at the north-east corner of each cell
        tlon, tlat, vlon, vlat = grid_coords(60, 30)
        np.testing.assert_allclose(tlon[:3], [3, 9, 15])
        np.testing.assert_allclose([tlat[0], tlat[-1]], [-87, 87])
        np.testing.assert_allclose([vlon[0], vlon[-1]], [6, 360])
        np.testing.assert_allclose([vlat[0], vlat[-1]], [-84, 84])
        self.assertEqual((len(tlon), len(tlat), len(vlon), len(vlat)), (60, 30, 60, 29))

    def test_constant_is_preserved(self):
        np.testing.assert_allclose(vel_to_temp(np.full((29, 60), 3.)), 3.)
        np.testing.assert_allclose(temp_to_vel(np.full((30, 60), 3.)), 3.)

    def test_vel_to_temp_averages_the_four_corners(self):
        rng = np.random.default_rng(0)
        f = rng.normal(size=(29, 60))
        g = vel_to_temp(f)
        self.assertEqual(g.shape, (30, 60))
        # temperature point (j,i) has velocity corners (j-1..j, i-1..i); i-1 wraps around
        for j, i in ((5, 7), (5, 0), (20, 59)):
            want = np.mean([f[jj, ii % 60] for jj in (j-1, j) for ii in (i-1, i)])
            self.assertAlmostEqual(g[j, i], want)
        # the polar rows only have one row of corners
        self.assertAlmostEqual(g[0, 7], np.mean([f[0, 6], f[0, 7]]))
        self.assertAlmostEqual(g[29, 7], np.mean([f[28, 6], f[28, 7]]))

    def test_linear_in_longitude_is_exact_in_the_interior(self):
        # a field linear in latitude and longitude (away from the wrap-around) averages back to itself
        tlon, tlat, vlon, vlat = grid_coords(60, 30)
        f = 2. * vlat[:, None] + 0.5 * vlon[None, :]
        g = vel_to_temp(f)
        want = 2. * tlat[1:-1, None] + 0.5 * tlon[None, 1:]
        np.testing.assert_allclose(g[1:-1, 1:], want, atol=1e-10)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.c = Context()
        self.model = get_model_class('bgrid_solo')(context=self.c, io_mode='offline')
        self.dir = tempfile.mkdtemp()
        self.time = datetime(2000, 1, 1)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write_state(self, **kwargs):
        m = self.model
        rng = np.random.default_rng(1)
        write_restart_file(m.filename(path=self.dir, time=self.time), m.nlon, m.nlat, m.nlev, m.model_day(self.time),
                           rng.normal(1e5, 1e3, (m.nlat, m.nlon)), rng.normal(250, 10, (m.nlev, m.nlat, m.nlon)),
                           rng.normal(0, 10, (m.nlev, m.nlat-1, m.nlon)), rng.normal(0, 10, (m.nlev, m.nlat-1, m.nlon)))

    def test_grid_and_variables(self):
        m = self.model
        self.assertEqual(m.grid.x.shape, (30, 60))
        self.assertEqual(set(m.variables), {'wind', 'temperature', 'ps'})
        self.assertEqual(len(m.variables['wind'].levels), 5)
        z = m.z_coords(name='temperature', k=0)
        self.assertEqual(z.shape, m.grid.x.shape)
        self.assertLess(z[0, 0], m.z_coords(name='temperature', k=4)[0, 0])  # level 0 is the top

    def test_model_day(self):
        self.assertEqual(self.model.model_day(self.model._time_ref + timedelta(hours=36)), 1.5)
        with self.assertRaises(AssertionError):
            self.model.model_day(self.model._time_ref - timedelta(days=1))

    def test_model_day_accepts_utc_aware_dates(self):
        aware = datetime(2001, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(self.model.model_day(aware), self.model.model_day(datetime(2001, 1, 1)))

    def test_grid_longitudes_increase_and_match_the_data(self):
        m = self.model
        x = m.grid.x[0]
        self.assertTrue(np.all(np.diff(x) > 0))
        self.assertEqual((x.min(), x.max()), (-177., 177.))
        # a temperature that depends on longitude only: the file has it in 0 ~ 360 order
        lon = (np.arange(m.nlon) + 0.5) * 360. / m.nlon
        t = np.tile(250. + 10. * np.sin(np.deg2rad(lon)), (m.nlev, m.nlat, 1))
        write_restart_file(m.filename(path=self.dir, time=self.time), m.nlon, m.nlat, m.nlev, m.model_day(self.time),
                           np.full((m.nlat, m.nlon), 1e5), t, np.zeros((m.nlev, m.nlat-1, m.nlon)), np.zeros((m.nlev, m.nlat-1, m.nlon)))
        got = m.read_var(path=self.dir, time=self.time, name='temperature', k=0)
        np.testing.assert_allclose(got[3], 250. + 10. * np.sin(np.deg2rad(x)))
        # and it is written back to the same columns
        new = got + 1.
        m.write_var(new, path=self.dir, time=self.time, name='temperature', k=0)
        with Dataset(m.filename(path=self.dir, time=self.time)) as f:
            np.testing.assert_allclose(f['t'][0, 0, 0, 3], t[0, 3] + 1.)
            np.testing.assert_allclose(f['t'][0, 0, 1, 3], t[1, 3])
        np.testing.assert_allclose(m.read_var(path=self.dir, time=self.time, name='temperature', k=0), new)

    def test_read_matches_file(self):
        self.write_state()
        m = self.model
        with Dataset(m.filename(path=self.dir, time=self.time)) as f:
            np.testing.assert_allclose(m.read_var(path=self.dir, time=self.time, name='ps'), m._to_grid(f['ps'][0, 0]))
            np.testing.assert_allclose(m.read_var(path=self.dir, time=self.time, name='temperature', k=3), m._to_grid(f['t'][0, 0, 3]))
            wind = m.read_var(path=self.dir, time=self.time, name='wind', k=2)
            np.testing.assert_allclose(wind[0], m._to_grid(vel_to_temp(f['u'][0, 0, 2])))
            np.testing.assert_allclose(wind[1], m._to_grid(vel_to_temp(f['v'][0, 0, 2])))

    def test_unchanged_write_leaves_the_file_alone(self):
        self.write_state()
        m = self.model
        fname = m.filename(path=self.dir, time=self.time)
        def contents():
            with Dataset(fname) as f:
                return {v: f[v][:].copy() for v in ('ps', 't', 'u', 'v')}
        before = contents()
        for name, k in (('ps', 0), ('temperature', 2), ('wind', 2)):
            m.write_var(m.read_var(path=self.dir, time=self.time, name=name, k=k), path=self.dir, time=self.time, name=name, k=k)
        after = contents()
        for v in before:
            np.testing.assert_array_equal(after[v], before[v])

    def test_written_wind_increment_is_seen_on_reading(self):
        self.write_state()
        m = self.model
        kw = dict(path=self.dir, time=self.time, name='wind', k=1)
        wind = m.read_var(**kw)
        other_level = m.read_var(**{**kw, 'k': 0})
        new = wind.copy()
        new[0] += 1.
        new[1] -= 2.
        m.write_var(new, **kw)
        got = m.read_var(**kw)
        # a uniform increment is carried exactly; the other levels are untouched
        np.testing.assert_allclose(got[0] - wind[0], 1., atol=1e-10)
        np.testing.assert_allclose(got[1] - wind[1], -2., atol=1e-10)
        np.testing.assert_array_equal(m.read_var(**{**kw, 'k': 0}), other_level)

    def test_write_scalar_fields(self):
        self.write_state()
        m = self.model
        new = np.full((m.nlat, m.nlon), 99000.)
        m.write_var(new, path=self.dir, time=self.time, name='ps')
        np.testing.assert_array_equal(m.read_var(path=self.dir, time=self.time, name='ps'), new)


@unittest.skipUnless(os.path.exists(EXE), "nedas_bgrid_advance not built")
class TestRun(unittest.TestCase):
    def setUp(self):
        self.c = Context()
        self.model = get_model_class('bgrid_solo')(context=self.c, io_mode='offline', spinup_hours=0)
        self.dir = tempfile.mkdtemp()
        self.time = datetime(2000, 1, 10)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_cold_start_is_at_rest_plus_perturbation(self):
        m = self.model
        m.init_perturb_sd = 0.5
        m.cold_start(m.filename(path=self.dir, time=self.time), self.time, seed=3)
        t = m.read_var(path=self.dir, time=self.time, name='temperature', k=2)
        self.assertAlmostEqual(t.mean(), 255., delta=0.1)
        self.assertAlmostEqual(t.std(), 0.5, delta=0.05)
        np.testing.assert_array_equal(m.read_var(path=self.dir, time=self.time, name='wind', k=2), 0.)
        np.testing.assert_array_equal(m.read_var(path=self.dir, time=self.time, name='ps'), 1e5)

    def test_run_advances_the_state_and_time(self):
        m = self.model
        m.cold_start(m.filename(path=self.dir, time=self.time), self.time, seed=3)
        m.run(path=self.dir, time=self.time, forecast_period=48)
        later = self.time + timedelta(hours=48)
        fname = m.filename(path=self.dir, time=later)
        self.assertTrue(os.path.exists(fname))
        with Dataset(fname) as f:
            self.assertEqual(f['time'][0], m.model_day(later))
        # the Held-Suarez forcing starts to drive winds out of the rest state
        self.assertGreater(np.abs(m.read_var(path=self.dir, time=later, name='wind', k=2)).max(), 0.1)

    def test_one_long_run_equals_chained_runs(self):
        m = self.model
        m.cold_start(m.filename(path=self.dir, time=self.time), self.time, seed=3)
        shutil.copy(m.filename(path=self.dir, time=self.time), os.path.join(self.dir, 'start.nc'))
        m.run(path=self.dir, time=self.time, forecast_period=48)
        m.run(path=self.dir, time=self.time + timedelta(hours=48), forecast_period=24)
        chained = m.read_var(path=self.dir, time=self.time + timedelta(hours=72), name='temperature', k=2)
        other = os.path.join(self.dir, 'b')
        os.makedirs(other)
        shutil.copy(os.path.join(self.dir, 'start.nc'), m.filename(path=other, time=self.time))
        m.run(path=other, time=self.time, forecast_period=72)
        np.testing.assert_allclose(m.read_var(path=other, time=self.time + timedelta(hours=72), name='temperature', k=2),
                                   chained, atol=1e-8)

    def test_forecast_period_must_be_a_multiple_of_the_time_step(self):
        m = self.model
        m.cold_start(m.filename(path=self.dir, time=self.time), self.time, seed=3)
        with self.assertRaises(AssertionError):
            m.run(path=self.dir, time=self.time, forecast_period=1.5)


if __name__ == '__main__':
    unittest.main()
