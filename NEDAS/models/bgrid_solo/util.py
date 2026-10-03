"""
Helpers for the DART bgrid_solo restart files and namelist.

The model state lives on two staggered B-grids (see DART's bgrid_solo model_mod):
temperature T and surface pressure ps at the cell centers (nlon x nlat), horizontal wind u,v
at the cell's north-east corners (nlon x nlat-1). NEDAS carries every variable on the
temperature grid, so the wind is averaged to it when read, and a change to it is mapped back
to the wind grid when written.
"""
import numpy as np
from netCDF4 import Dataset


def grid_coords(nlon: int, nlat: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """longitudes, latitudes (degrees) of the temperature grid and of the velocity grid"""
    tlon = (np.arange(nlon) + 0.5) * 360. / nlon
    tlat = -90. + (np.arange(nlat) + 0.5) * 180. / nlat
    vlon = (np.arange(nlon) + 1.0) * 360. / nlon
    vlat = -90. + (np.arange(nlat - 1) + 1.0) * 180. / nlat
    return tlon, tlat, vlon, vlat


def vel_to_temp(f: np.ndarray) -> np.ndarray:
    """
    Average a field on the velocity grid (nlat-1, nlon) to the temperature grid (nlat, nlon).

    Velocity point (j,i) is the north-east corner of temperature cell (j,i), so the temperature
    point (j,i) is surrounded by velocity points (j-1..j, i-1..i). The two polar rows only have
    the two velocity points on their one side.
    """
    nj = f.shape[0]
    g = np.zeros((nj + 1, f.shape[1]))
    n = np.zeros_like(g)
    for dj in (0, 1):
        for di in (0, 1):
            g[dj:dj + nj] += np.roll(f, di, axis=1)
            n[dj:dj + nj] += 1
    return g / n


def temp_to_vel(g: np.ndarray) -> np.ndarray:
    """Average a field on the temperature grid to the velocity grid (each corner has 4 neighbors)"""
    a = 0.5 * (g[:-1] + g[1:])
    return 0.5 * (a + np.roll(a, -1, axis=1))


def write_restart_file(filename: str, nlon: int, nlat: int, nlev: int, day: float,
                       ps: np.ndarray, t: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    """Write a state in the layout of DART's bgrid_solo perfect_input.nc (also its model template)"""
    tlon, tlat, vlon, vlat = grid_coords(nlon, nlat)
    with Dataset(filename, 'w') as f:
        f.createDimension('member', 1)
        f.createDimension('metadatalength', 32)
        f.createDimension('TmpI', nlon)
        f.createDimension('TmpJ', nlat)
        f.createDimension('VelI', nlon)
        f.createDimension('VelJ', nlat - 1)
        f.createDimension('lev', nlev)
        f.createDimension('time', None)
        f.title = 'bgrid_solo state written by NEDAS'
        f.model = 'FMS_Bgrid'
        for name, vals, longname, axis, units in (
                ('TmpI', tlon, 'longitude', 'X', 'degrees_east'),
                ('TmpJ', tlat, 'latitude', 'Y', 'degrees_north'),
                ('VelI', vlon, 'longitude', 'X', 'degrees_east'),
                ('VelJ', vlat, 'latitude', 'Y', 'degrees_north')):
            x = f.createVariable(name, 'f8', (name,))
            x.long_name, x.cartesian_axis, x.units = longname, axis, units
            x[:] = vals
        x = f.createVariable('lev', 'i4', ('lev',))
        x.long_name, x.cartesian_axis, x.units, x.positive = 'level', 'Z', 'hPa', 'down'
        x[:] = np.arange(1, nlev + 1)
        x = f.createVariable('MemberMetadata', 'S1', ('member', 'metadatalength'))
        x.long_name = 'description of each member'
        x = f.createVariable('time', 'f8', ('time',))
        x.long_name, x.axis, x.cartesian_axis, x.calendar, x.units = \
            'valid time of the model state', 'T', 'T', 'none', 'days'
        x[0] = day
        for name, dims, longname, units, vals in (
                ('ps', ('TmpJ', 'TmpI'), 'surface pressure', 'Pa', ps),
                ('t', ('lev', 'TmpJ', 'TmpI'), 'temperature', 'degrees Kelvin', t),
                ('u', ('lev', 'VelJ', 'VelI'), 'zonal wind component', 'm/s', u),
                ('v', ('lev', 'VelJ', 'VelI'), 'meridional wind component', 'm/s', v)):
            x = f.createVariable(name, 'f8', ('time', 'member') + dims)
            x.long_name, x.units = longname, units
            x[0, 0] = vals


def input_nml(model, template: str, ic_file: str, ud_file: str, advance_seconds: int=0,
              cold_start: bool=False, init_days: int=0) -> str:
    """The input.nml text for nedas_bgrid_advance, with the settings of a BgridSoloModel"""
    b = lambda x: '.true.' if x else '.false.'
    return f"""&utilities_nml
   TERMLEVEL = 2,
   logfilename = 'dart_log.out',
   nmlfilename = 'dart_log.nml',
   write_nml   = 'none'
   /
&nedas_bgrid_advance_nml
   ic_file         = '{ic_file}',
   ud_file         = '{ud_file}',
   advance_days    = {advance_seconds // 86400},
   advance_seconds = {advance_seconds % 86400},
   cold_start      = {b(cold_start)},
   init_days       = {init_days},
   init_seconds    = 0
   /
&model_nml
   current_time = 0, 0, 0, 0
   override = .false.,
   dt_atmos = {model.dt_atmos},
   days = 0, hours = 0, minutes = 0, seconds = 0,
   noise_sd = 0.0,
   dt_bias  = -1,
   state_variables = 'ps', 'QTY_SURFACE_PRESSURE',
                     't',  'QTY_TEMPERATURE',
                     'u',  'QTY_U_WIND_COMPONENT',
                     'v',  'QTY_V_WIND_COMPONENT',
   template_file = '{template}'
   /
&fms_nml
   domains_stack_size = 90000
   /
&bgrid_cold_start_nml
   nlon = {model.nlon},
   nlat = {model.nlat},
   nlev = {model.nlev},
   equal_vert_spacing = .true.
   /
&hs_forcing_nml
   delh = {model.delh}, t_zero = {model.t_zero}, t_strat = {model.t_strat}, delv = {model.delv},
   eps = 0., ka = {model.ka}, ks = {model.ks}, kf = {model.kf}, sigma_b = {model.sigma_b},
   do_conserve_energy = .false.
   /
&bgrid_core_driver_nml
   damp_coeff_wind   = {model.damp_coeff_wind},
   damp_coeff_temp   = {model.damp_coeff_temp},
   damp_coeff_tracer = 0.10,
   advec_order_wind   = {model.advec_order_wind},
   advec_order_temp   = {model.advec_order_temp},
   advec_order_tracer = 2,
   num_sponge_levels = 1,
   sponge_coeff_wind   = 1.00,
   sponge_coeff_temp   = 1.00,
   sponge_coeff_tracer = 1.00,
   num_fill_pass = 2,
   decomp = 0,0,
   num_adjust_dt = 3,
   num_advec_dt  = 3,
   halo = 1,
   do_conserve_energy = .false.
   /
&bgrid_integrals_nml
   file_name  = 'dynam_integral.out',
   time_units = 'days',
   output_interval = 1.00
   /
&obs_kind_nml
   /
&ensemble_manager_nml
   /
&state_vector_io_nml
   /
&atmosphere_nml
   /
&topography_nml
   /
&gaussian_topog_nml
   /
&location_nml
   horiz_dist_only             = .true.,
   vert_normalization_pressure = 100000.0,
   vert_normalization_height   = 10000.0,
   vert_normalization_level    = 20.0,
   approximate_distance        = .true.,
   nlon                        = 71,
   nlat                        = 36,
   output_box_info             = .false.
   /
"""
