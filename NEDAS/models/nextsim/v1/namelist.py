import os

def value_str(value):
    ##convert values to string in namelist
    if isinstance(value, bool):
        if value:
            vstr = 'true'
        else:
            vstr = 'false'
    elif isinstance(value, str):
        vstr = value
    else:
        vstr = f'{value}'
    return vstr

def namelist(m, time, forecast_period, run_dir='.'):
    """Generate namelist for nextsim v1 model
    Input:
    -m: Model class object with model configurations
    -time: start time (datetime obj)
    -forecast_period: period (hour) to run the forecast
    -run_dir: directory where the model runtime files are stored
    """
    ##start building the namelist content
    nmlstr = "[mesh]\n"
    nmlstr += "filename="+value_str(m.msh_filename)+"\n"
    nmlstr += "[setup]\n"
    nmlstr += "ice-type="+value_str(m.ice_type)+"\n"
    nmlstr += "ocean-type="+value_str(m.ocean_type)+"\n"
    nmlstr += "atmosphere-type="+value_str(m.atmosphere_type)+"\n"
    nmlstr += "bathymetry-type="+value_str(m.bathymetry_type)+"\n"
    nmlstr += "use_assimilation=false\n"
    nmlstr += "dynamics-type="+value_str(m.dynamics_type)+"\n"
    nmlstr += "atmospheric_forcing_input_path="+os.path.join('data',m.atmos_forcing_path)+"\n"
    nmlstr += "oceanic_forcing_input_path="+os.path.join('data',m.ocean_forcing_path)+"\n"
    nmlstr += "[simul]\n"
    nmlstr += "spinup_duration=0\n"
    nmlstr += "timestep="+value_str(int(m.timestep))+"\n"
    nmlstr += "time_init="+time.strftime('%Y-%m-%d %H:%M:%S')+"\n"
    nmlstr += "duration="+f"{forecast_period / 24}"+"\n"
    nmlstr += "[thermo]\n"
    nmlstr += "use_assim_flux=false\n"
    nmlstr += "assim_flux_exponent=4\n"
    nmlstr += "diffusivity_sss="+value_str(m.diffusivity_sss)+"\n"
    nmlstr += "diffusivity_sst="+value_str(m.diffusivity_sst)+"\n"
    nmlstr += "ocean_nudge_timeS_days="+value_str(m.ocean_nudge_timeS)+"\n"
    nmlstr += "ocean_nudge_timeT_days="+value_str(m.ocean_nudge_timeT)+"\n"
    nmlstr += "freezingpoint-type="+m.freezingpoint_type+"\n"
    nmlstr += "h_young_max="+value_str(m.h_young_max)+"\n"
    nmlstr += "alb_ice="+value_str(m.alb_ice)+"\n"
    nmlstr += "alb_sn="+value_str(m.alb_sn)+"\n"
    nmlstr += "albedoW="+value_str(m.albedoW)+"\n"
    nmlstr += "melt_type="+value_str(m.melt_type)+"\n"
    nmlstr += "PhiM="+value_str(m.PhiM)+"\n"
    nmlstr += "snow_cond="+value_str(m.snow_cond)+"\n"
    nmlstr += "[dynamics]\n"
    nmlstr += "time_relaxation_damage="+value_str(m.time_relaxation_damage)+"\n"
    nmlstr += "compression_factor="+value_str(m.compression_factor)+"\n"
    nmlstr += "C_lab="+value_str(m.cohesion)+"\n"
    nmlstr += "substeps="+value_str(m.substeps)+"\n"
    nmlstr += "use_temperature_dependent_healing="+value_str(m.use_temperature_dependent_healing)+"\n"
    nmlstr += "ECMWF_quad_drag_coef_air="+value_str(m.ECMWF_quad_drag_coef_air)+"\n"
    nmlstr += "quad_drag_coef_water="+value_str(m.quad_drag_coef_water)+"\n"
    nmlstr += "Lemieux_basal_k1="+value_str(m.Lemieux_basal_k1)+"\n"
    nmlstr += "compaction_param="+value_str(m.compaction_param)+"\n"
    nmlstr += "oceanic_turning_angle="+value_str(m.oceanic_turning_angle)+"\n"
    nmlstr += "[age]\n"
    nmlstr += "include_young_ice="+value_str(m.include_young_ice)+"\n"
    nmlstr += "[restart]\n"
    nmlstr += "start_from_restart="+value_str(m.start_from_restart)+"\n"
    nmlstr += "type="+value_str(m.restart_type)+"\n"
    nmlstr += "write_initial_restart="+value_str(m.write_initial_restart)+"\n"
    nmlstr += "write_interval_restart="+value_str(m.write_interval_restart)+"\n"
    nmlstr += "output_interval="+f"{m.restart_dt / 24}"+"\n"
    nmlstr += "check_restart="+value_str(m.check_restart)+"\n"
    nmlstr += "input_path="+value_str(m.restart_input_path)+"\n"
    nmlstr += "basename="+time.strftime('%Y%m%dT%H%M%SZ')+"\n"
    nmlstr += "[output]\n"
    nmlstr += "output_per_day="+f"{int(24 / m.restart_dt)}"+"\n"
    nmlstr += "datetime_in_filename="+value_str(m.datetime_in_filename)+"\n"
    nmlstr += "exporter_path="+value_str(m.exporter_path)+"\n"
    nmlstr += "export_before_regrid="+value_str(m.export_before_regrid)+"\n"
    nmlstr += "export_after_regrid="+value_str(m.export_after_regrid)+"\n"
    nmlstr += "[moorings]\n"
    nmlstr += "use_moorings="+value_str(m.use_moorings)+"\n"
    nmlstr += "grid_type="+value_str(m.mooring_grid_type)+"\n"
    nmlstr += "spacing="+value_str(m.mooring_spacing)+"\n"
    nmlstr += "output_timestep="+f"{m.restart_dt / 24}"+"\n"
    for varname in m.mooring_variables:
        nmlstr += "variables="+varname+"\n"
    nmlstr += "[drifters]\n"
    nmlstr += "use_osisaf_drifters="+value_str(m.use_osisaf_drifters)+"\n"
    nmlstr += "use_sidfex_drifters="+value_str(m.use_sidfex_drifters)+"\n"
    nmlstr += "sidfex_drifters_output_time_step="+f"{m.sidfex_output_dt / 24}"+"\n"
    nmlstr += "sidfex_ignore_restart="+value_str(m.sidfex_ignore_restart)+"\n"
    nmlstr += "[debugging]\n"
    nmlstr += "check_fields_fast="+value_str(m.check_fields_fast)+"\n"

    ##write the namelist to nextsim.cfg file
    with open(os.path.join(run_dir, 'nextsim.cfg'), 'wt') as f:
        f.write(nmlstr)

