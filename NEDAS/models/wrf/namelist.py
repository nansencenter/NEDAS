import os
import numpy as np
from NEDAS.utils.conversion import dt1h

def bool_str(value):
    if value:
        return '.true.'
    else:
        return '.false.'

def namelist(m, time, forecast_period, run_dir='.'):

    next_time = time + forecast_period * dt1h

    ###wps namelist.wps
    """
    nmlstr = "&share\n"
    nmlstr += "wrf_core = 'ARW',"
    nmlstr += "max_dom = 2,
    nmlstr += "start_date = '2000-01-24_12:00:00','2000-01-24_12:00:00',
    nmlstr += "end_date   = '2000-01-25_12:00:00','2000-01-24_12:00:00',
    nmlstr += "interval_seconds = 21600
    nmlstr += "active_grid = .true., .true.,
    nmlstr += "subgrid_ratio_x = 1
    nmlstr += "subgrid_ratio_y = 1
    nmlstr += "io_form_geogrid = 2,
    nmlstr += "opt_output_from_geogrid_path = './',
    nmlstr += "debug_level = 0
    nmlstr += "/\n\n"

    nmlstr += "&geogrid\n"
    nmlstr += "parent_id         =   1,   1,
    nmlstr += "parent_grid_ratio =   1,   3,
    nmlstr += "i_parent_start    =   1,  31,
    nmlstr += "j_parent_start    =   1,  17,
    nmlstr += "s_we              =   1,   1,
    nmlstr += "e_we              =  74, 112,
    nmlstr += "s_sn              =   1,   1,
    nmlstr += "e_sn              =  61,  97,
    nmlstr += "geog_data_res = 'default','default',
    nmlstr += "dx        = 30000,
    nmlstr += "dy        = 30000,
    nmlstr += "map_proj  = 'lambert',
    nmlstr += "ref_lat   =  34.83,
    nmlstr += "ref_lon   = -81.03,
    nmlstr += "ref_x     =  37.0,
    nmlstr += "ref_y     =  30.5,
    nmlstr += "truelat1  =  30.0,
    nmlstr += "truelat2  =  60.0,
    nmlstr += "stand_lon = -98.0,
    nmlstr += "geog_data_path = '/glade/work/wrfhelp/WPS_GEOG/'
    nmlstr += "opt_geogrid_tbl_path = 'geogrid/'
    nmlstr += "/
    nmlstr += "geog_data_res     = 'modis_lakes+10m','modis_lakes+2m',
    nmlstr += "geog_data_res     = 'usgs_lakes+10m','usgs_lakes+2m',

    nmlstr += "&ungrib
    nmlstr += "out_format = 'WPS',
    nmlstr += "prefix     = 'FILE',
    nmlstr += "ec_rec_len = 26214508,
    nmlstr += "pmin = 100.
    nmlstr += "/

    nmlstr += "&metgrid
    nmlstr += "fg_name         = 'FILE'
    nmlstr += "constants_name  = './TAVGSFC'
    nmlstr += "io_form_metgrid = 2, 
    nmlstr += "opt_output_from_metgrid_path = './',
    nmlstr += "opt_metgrid_tbl_path         = 'metgrid/',
    nmlstr += "process_only_bdy = 5,
    nmlstr += "/

    nmlstr += "&mod_levs
    nmlstr += "press_pa = 201300 , 200100 , 100000 , 
    nmlstr += "/\n\n"

    with open(os.path.join(run_dir, 'namelist.wps'), 'wt') as f:
        f.write(nmlstr)


    ##wrf namelist.input
    nmlstr = f"&time_control\n"
    nmlstr += f"run_days         = {int(forecast_period/24)},\n"
    nmlstr += f"run_hours        = {int(forecast_period%24)},\n"
    nmlstr += f"run_minutes      = {int(np.round(forecast_period - int(forecast_period)))},\n"
    nmlstr += f"run_seconds      = 0,\n"
    nmlstr += f"start_year       = 2019, 2019,
    nmlstr += f"start_month      = 09,   09, 
    nmlstr += f"start_day        = 04,   04,
    nmlstr += f"start_hour       = 12,   12,
    nmlstr += f"end_year         = 2019, 2019,
    nmlstr += f"end_month        = 09,   09,
    nmlstr += f"end_day          = 06,   06,
    nmlstr += f"end_hour         = 00,   00,
    nmlstr += f"interval_seconds = 10800
    nmlstr += f"input_from_file  = .true.,.true.,\n"
    nmlstr += f"history_interval = 60,  60,\n"
    nmlstr += f"frames_per_outfile = 1, 1,\n"
    nmlstr += f"restart            = .false.,\n"
    nmlstr += f"restart_interval   = 7200,\n"
    nmlstr += f"io_form_history    = 2,\n"
    nmlstr += f"io_form_restart    = 2,\n"
    nmlstr += f"io_form_input      = 2,\n"
    nmlstr += f"io_form_boundary   = 2,\n"
    nmlstr += f"/\n"

    nmlstr += "&domains
    nmlstr += "time_step                           = 90,
    nmlstr += "time_step_fract_num                 = 0,
    nmlstr += "time_step_fract_den                 = 1,
    nmlstr += "max_dom                             = 2,
    nmlstr += "e_we                                = 150,    220,
    nmlstr += "e_sn                                = 130,    214,
    nmlstr += "e_vert                              = 45,     45,
    nmlstr += "dzstretch_s                         = 1.1
    nmlstr += "p_top_requested                     = 5000,
    nmlstr += "num_metgrid_levels                  = 34,
    nmlstr += "num_metgrid_soil_levels             = 4,
    nmlstr += "dx                                  = 15000,
    nmlstr += "dy                                  = 15000,
    nmlstr += "grid_id                             = 1,     2,
    nmlstr += "parent_id                           = 0,     1,
    nmlstr += "i_parent_start                      = 1,     53,
    nmlstr += "j_parent_start                      = 1,     25,
    nmlstr += "parent_grid_ratio                   = 1,     3,
    nmlstr += "parent_time_step_ratio              = 1,     3,
    nmlstr += "feedback                            = 1,
    nmlstr += "smooth_option                       = 0
    nmlstr += "/

    nmlstr += "&physics
    nmlstr += "physics_suite                       = 'CONUS'
    nmlstr += "mp_physics                          = -1,    -1,
    nmlstr += "cu_physics                          = -1,    -1,
    nmlstr += "ra_lw_physics                       = -1,    -1,
    nmlstr += "ra_sw_physics                       = -1,    -1,
    nmlstr += "bl_pbl_physics                      = -1,    -1,
    nmlstr += "sf_sfclay_physics                   = -1,    -1,
    nmlstr += "sf_surface_physics                  = -1,    -1,
    nmlstr += "radt                                = 15,    15,
    nmlstr += "bldt                                = 0,     0,
    nmlstr += "cudt                                = 0,     0,
    nmlstr += "icloud                              = 1,
    nmlstr += "num_land_cat                        = 21,
    nmlstr += "sf_urban_physics                    = 0,     0,
    nmlstr += "fractional_seaice                   = 1,
    nmlstr += "/

    nmlstr += "&fdda
    nmlstr += "/

    nmlstr += "&dynamics
    nmlstr += "hybrid_opt                          = 2, 
    nmlstr += "w_damping                           = 0,
    nmlstr += "diff_opt                            = 2,      2,
    nmlstr += "km_opt                              = 4,      4,
    nmlstr += "diff_6th_opt                        = 0,      0,
    nmlstr += "iff_6th_factor                     = 0.12,   0.12,
    nmlstr += "base_temp                           = 290.
    nmlstr += "damp_opt                            = 3,
    nmlstr += "zdamp                               = 5000.,  5000.,
    nmlstr += "dampcoef                            = 0.2,    0.2,
    nmlstr += "khdif                               = 0,      0,
    nmlstr += "kvdif                               = 0,      0,
    nmlstr += "non_hydrostatic                     = .true., .true.,
    nmlstr += "moist_adv_opt                       = 1,      1,
    nmlstr += "scalar_adv_opt                      = 1,      1,
    nmlstr += "gwd_opt                             = 1,      0,
    nmlstr += "/

    nmlstr += "&bdy_control
    nmlstr += "spec_bdy_width                      = 5,
    nmlstr += "specified                           = .true.
    nmlstr += "/

    nmlstr += "&grib2
    nmlstr += "/

    nmlstr += "&namelist_quilt
    nmlstr += "nio_tasks_per_group = 0,
    nmlstr += "nio_groups = 1,
    nmlstr += "/

    with open(os.path.join(run_dir, 'namelist.input'), 'wt') as f:
        f.write(nmlstr)
    """