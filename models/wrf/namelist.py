import os


def namelist_wrf(m, run_dir='.'):
    nmlstr = "&time_control\n"
    run_days                            = 0,
    run_hours                           = 36,
    run_minutes                         = 0,
    run_seconds                         = 0,
    start_year                          = 2019, 2019,
    start_month                         = 09,   09, 
    start_day                           = 04,   04,
    start_hour                          = 12,   12,
    end_year                            = 2019, 2019,
    end_month                           = 09,   09,
    end_day                             = 06,   06,
    end_hour                            = 00,   00,
    interval_seconds                    = 10800
    input_from_file                     = .true.,.true.,
    history_interval                    = 60,  60,
    frames_per_outfile                  = 1, 1,
    restart                             = .false.,
    restart_interval                    = 7200,
    io_form_history                     = 2
    io_form_restart                     = 2
    io_form_input                       = 2
    io_form_boundary                    = 2
    /

    &domains
    time_step                           = 90,
    time_step_fract_num                 = 0,
    time_step_fract_den                 = 1,
    max_dom                             = 2,
    e_we                                = 150,    220,
    e_sn                                = 130,    214,
    e_vert                              = 45,     45,
    dzstretch_s                         = 1.1
    p_top_requested                     = 5000,
    num_metgrid_levels                  = 34,
    num_metgrid_soil_levels             = 4,
    dx                                  = 15000,
    dy                                  = 15000,
    grid_id                             = 1,     2,
    parent_id                           = 0,     1,
    i_parent_start                      = 1,     53,
    j_parent_start                      = 1,     25,
    parent_grid_ratio                   = 1,     3,
    parent_time_step_ratio              = 1,     3,
    feedback                            = 1,
    smooth_option                       = 0
    /

    &physics
    physics_suite                       = 'CONUS'
    mp_physics                          = -1,    -1,
    cu_physics                          = -1,    -1,
    ra_lw_physics                       = -1,    -1,
    ra_sw_physics                       = -1,    -1,
    bl_pbl_physics                      = -1,    -1,
    sf_sfclay_physics                   = -1,    -1,
    sf_surface_physics                  = -1,    -1,
    radt                                = 15,    15,
    bldt                                = 0,     0,
    cudt                                = 0,     0,
    icloud                              = 1,
    num_land_cat                        = 21,
    sf_urban_physics                    = 0,     0,
    fractional_seaice                   = 1,
    /

    &fdda
    /

    &dynamics
    hybrid_opt                          = 2, 
    w_damping                           = 0,
    diff_opt                            = 2,      2,
    km_opt                              = 4,      4,
    diff_6th_opt                        = 0,      0,
    diff_6th_factor                     = 0.12,   0.12,
    base_temp                           = 290.
    damp_opt                            = 3,
    zdamp                               = 5000.,  5000.,
    dampcoef                            = 0.2,    0.2,
    khdif                               = 0,      0,
    kvdif                               = 0,      0,
    non_hydrostatic                     = .true., .true.,
    moist_adv_opt                       = 1,      1,
    scalar_adv_opt                      = 1,      1,
    gwd_opt                             = 1,      0,
    /

    &bdy_control
    spec_bdy_width                      = 5,
    specified                           = .true.
    /

    &grib2
    /

    &namelist_quilt
    nio_tasks_per_group = 0,
    nio_groups = 1,
    /

    with open(os.path.join(run_dir, 'namelist.input'), 'wt') as f:
        f.write(nmlstr)


def namelist_wps(m, run_dir="."):
    

    &share
    wrf_core = 'ARW',
    max_dom = 2,
    start_date = '2006-08-16_12:00:00','2006-08-16_12:00:00',
    end_date   = '2006-08-16_18:00:00','2006-08-16_12:00:00',
    interval_seconds = 21600
    active_grid = .true., .true.,
    subgrid_ratio_x = 1
    subgrid_ratio_y = 1
    io_form_geogrid = 2,
    opt_output_from_geogrid_path = './',
    debug_level = 0
    /
    start_date = '2000-01-24_12:00:00','2000-01-24_12:00:00',
    end_date   = '2000-01-25_12:00:00','2000-01-24_12:00:00',
    start_year   = 2006, 2006,
    start_month  =   08,   08,
    start_day    =   16,   16,
    start_hour   =   12,   12,
    start_minute =   00,   00,
    start_second =   00,   00,
    end_year     = 2006, 2006,
    end_month    =   08,   08,
    end_day      =   16,   16,
    end_hour     =   18,   12,
    end_minute   =   00,   00,
    end_second   =   00,   00,

    &geogrid
    parent_id         =   1,   1,
    parent_grid_ratio =   1,   3,
    i_parent_start    =   1,  31,
    j_parent_start    =   1,  17,
    s_we              =   1,   1,
    e_we              =  74, 112,
    s_sn              =   1,   1,
    e_sn              =  61,  97,
    !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! The default datasets used to produce the MAXSNOALB and ALBEDO12M
    ! fields have changed in WPS v4.0. These fields are now interpolated
    ! from MODIS-based datasets.
    !
    ! To match the output given by the default namelist.wps in WPS v3.9.1,
    ! the following setting for geog_data_res may be used:
    !
    ! geog_data_res = 'maxsnowalb_ncep+albedo_ncep+default', 'maxsnowalb_ncep+albedo_ncep+default',
    !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !
    geog_data_res = 'default','default',
    dx        = 30000,
    dy        = 30000,
    map_proj  = 'lambert',
    ref_lat   =  34.83,
    ref_lon   = -81.03,
    ref_x     =  37.0,
    ref_y     =  30.5,
    truelat1  =  30.0,
    truelat2  =  60.0,
    stand_lon = -98.0,
    geog_data_path = '/glade/work/wrfhelp/WPS_GEOG/'
    opt_geogrid_tbl_path = 'geogrid/'
    /
    geog_data_res     = 'modis_lakes+10m','modis_lakes+2m',
    geog_data_res     = 'usgs_lakes+10m','usgs_lakes+2m',

    &ungrib
    out_format = 'WPS',
    prefix     = 'FILE',
    ec_rec_len = 26214508,
    pmin = 100.
    /

    &metgrid
    fg_name         = 'FILE'
    constants_name  = './TAVGSFC'
    io_form_metgrid = 2, 
    opt_output_from_metgrid_path = './',
    opt_metgrid_tbl_path         = 'metgrid/',
    process_only_bdy = 5,
    /

    &mod_levs
    press_pa = 201300 , 200100 , 100000 , 
                95000 ,  90000 , 
                85000 ,  80000 , 
                75000 ,  70000 , 
                65000 ,  60000 , 
                55000 ,  50000 , 
                45000 ,  40000 , 
                35000 ,  30000 , 
                25000 ,  20000 , 
                15000 ,  10000 , 
                5000 ,   1000
    /

    &plotfmt
    ix = 100
    jx = 100
    ioff = 30
    joff = 30
    /
