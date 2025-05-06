import os

def value_str(value):
    ##convert values to string in namelist
    if isinstance(value, bool):
        if value:
            vstr = 'T'
        else:
            vstr = 'F'
    elif isinstance(value, str):
        vstr = "'"+value+"'"
    else:
        vstr = f'{value}'
    return vstr

def namelist(conf_dict, time, forecast_period, psi_init_type, member=0, dt_ratio=1, run_dir='.'):
    """Generate namelist for qg model
    Input:
    - m: Model class object with model configurations
    - dt_ratio: factor to multiply dt with
    """
    # model time step
    dt = conf_dict['dt'] * dt_ratio
    forecast_days = float(forecast_period) / 24
    dt_days = dt / conf_dict['tscale']
    # number of time steps to run
    total_counts = forecast_days / dt_days

    #random number based on member and time
    idum = 1000*member + 100*time.month + 10*time.day + time.hour

    #initialize energy only if at creation of initial condition
    #while during cycling, read the input from previous cycle and don't normalize energy
    if psi_init_type == 'read':
        initialize_energy = False
    else:
        initialize_energy = True

    ##start building the namelist content
    nmlstr = " &run_params\n"
    nmlstr += " kmax = "+value_str(conf_dict['kmax'])+"\n"
    nmlstr += " nz = "+value_str(conf_dict['nz'])+"\n"
    nmlstr += " F = "+value_str(conf_dict['F'])+"\n"
    nmlstr += " beta = "+value_str(conf_dict['beta'])+"\n"
    nmlstr += " adapt_dt = "+value_str(conf_dict['adapt_dt'])+"\n"
    nmlstr += " dt = "+value_str(dt)+"\n"
    nmlstr += " psi_init_file = 'input'\n"
    nmlstr += " psi_init_type = "+value_str(psi_init_type)+"\n"
    nmlstr += " initialize_energy = "+value_str(initialize_energy)+"\n"
    nmlstr += " e_o = "+value_str(conf_dict['e_o'])+"\n"
    nmlstr += " k_o = "+value_str(conf_dict['k_o'])+"\n"
    nmlstr += " delk = "+value_str(conf_dict['delk'])+"\n"
    nmlstr += " m_o = "+value_str(conf_dict['m_o'])+"\n"
    nmlstr += " z_o = "+value_str(conf_dict['z_o'])+"\n"
    nmlstr += " strat_type = "+value_str(conf_dict['strat_type'])+"\n"
    nmlstr += " deltc = "+value_str(conf_dict['deltc'])+"\n"
    nmlstr += " ubar_type = "+value_str(conf_dict['ubar_type'])+"\n"
    nmlstr += " delu = "+value_str(conf_dict['delu'])+"\n"
    nmlstr += " uscale = "+value_str(conf_dict['uscale'])+"\n"
    nmlstr += " use_topo = "+value_str(conf_dict['use_topo'])+"\n"
    nmlstr += " topo_type = "+value_str(conf_dict['topo_type'])+"\n"
    nmlstr += " k_o_topo = "+value_str(conf_dict['k_o_topo'])+"\n"
    nmlstr += " del_topo = "+value_str(conf_dict['del_topo'])+"\n"
    nmlstr += " use_forcing = "+value_str(conf_dict['use_forcing'])+"\n"
    nmlstr += " norm_forcing = "+value_str(conf_dict['norm_forcing'])+"\n"
    nmlstr += " forc_coef = "+value_str(conf_dict['forc_coef'])+"\n"
    nmlstr += " forc_corr = "+value_str(conf_dict['forc_corr'])+"\n"
    nmlstr += " kf_min = "+value_str(conf_dict['kf_min'])+"\n"
    nmlstr += " kf_max = "+value_str(conf_dict['kf_max'])+"\n"
    nmlstr += " filter_type = "+value_str(conf_dict['filter_type'])+"\n"
    nmlstr += " filter_exp = "+value_str(conf_dict['filter_exp'])+"\n"
    nmlstr += " filt_tune = "+value_str(conf_dict['filt_tune'])+"\n"
    nmlstr += " k_cut = "+value_str(conf_dict['k_cut'])+"\n"
    nmlstr += " rho_slope = "+value_str(conf_dict['rho_slope'])+"\n"
    nmlstr += " bot_drag = "+value_str(conf_dict['bot_drag'])+"\n"
    nmlstr += " therm_drag = "+value_str(conf_dict['therm_drag'])+"\n"
    nmlstr += " idum = "+value_str(idum)+"\n"
    nmlstr += " total_counts = "+value_str(total_counts)+"\n"
    nmlstr += " write_step = "+value_str(total_counts)+"\n"
    nmlstr += " diag1_step = "+value_str(total_counts)+"\n"
    nmlstr += " diag2_step = "+value_str(total_counts)+"\n"
    nmlstr += " /"

    ##write the namelist to input.nml file
    with open(os.path.join(run_dir, 'input.nml'), 'wt') as f:
        f.write(nmlstr)

