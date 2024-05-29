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


def namelist(m, run_dir):
    """Generate namelist for qg model
    Input:
    - m: Model class object with model configurations
    """
    ##start building the namelist content
    nmlstr = " &run_params\n"
    nmlstr += " kmax = "+value_str(m.kmax)+"\n"
    nmlstr += " nz = "+value_str(m.nz)+"\n"
    nmlstr += " F = "+value_str(m.F)+"\n"
    nmlstr += " beta = "+value_str(m.beta)+"\n"
    nmlstr += " adapt_dt = "+value_str(m.adapt_dt)+"\n"
    nmlstr += " dt = "+value_str(m.dt)+"\n"
    nmlstr += " psi_init_file = "+value_str(m.psi_init_file)+"\n"
    nmlstr += " psi_init_type = "+value_str(m.psi_init_type)+"\n"
    nmlstr += " initialize_energy = "+value_str(m.initialize_energy)+"\n"
    nmlstr += " e_o = "+value_str(m.e_o)+"\n"
    nmlstr += " k_o = "+value_str(m.k_o)+"\n"
    nmlstr += " delk = "+value_str(m.delk)+"\n"
    nmlstr += " m_o = "+value_str(m.m_o)+"\n"
    nmlstr += " z_o = "+value_str(m.z_o)+"\n"
    nmlstr += " strat_type = "+value_str(m.strat_type)+"\n"
    nmlstr += " deltc = "+value_str(m.deltc)+"\n"
    nmlstr += " ubar_type = "+value_str(m.ubar_type)+"\n"
    nmlstr += " delu = "+value_str(m.delu)+"\n"
    nmlstr += " uscale = "+value_str(m.uscale)+"\n"
    nmlstr += " use_topo = "+value_str(m.use_topo)+"\n"
    nmlstr += " topo_type = "+value_str(m.topo_type)+"\n"
    nmlstr += " k_o_topo = "+value_str(m.k_o_topo)+"\n"
    nmlstr += " del_topo = "+value_str(m.del_topo)+"\n"
    nmlstr += " use_forcing = "+value_str(m.use_forcing)+"\n"
    nmlstr += " norm_forcing = "+value_str(m.norm_forcing)+"\n"
    nmlstr += " forc_coef = "+value_str(m.forc_coef)+"\n"
    nmlstr += " forc_corr = "+value_str(m.forc_corr)+"\n"
    nmlstr += " kf_min = "+value_str(m.kf_min)+"\n"
    nmlstr += " kf_max = "+value_str(m.kf_max)+"\n"
    nmlstr += " filter_type = "+value_str(m.filter_type)+"\n"
    nmlstr += " filter_exp = "+value_str(m.filter_exp)+"\n"
    nmlstr += " filt_tune = "+value_str(m.filt_tune)+"\n"
    nmlstr += " k_cut = "+value_str(m.k_cut)+"\n"
    nmlstr += " rho_slope = "+value_str(m.rho_slope)+"\n"
    nmlstr += " bot_drag = "+value_str(m.bot_drag)+"\n"
    nmlstr += " therm_drag = "+value_str(m.therm_drag)+"\n"
    nmlstr += " idum = "+value_str(m.idum)+"\n"
    nmlstr += " total_counts = "+value_str(m.total_counts)+"\n"
    nmlstr += " write_step = "+value_str(m.write_step)+"\n"
    nmlstr += " diag1_step = "+value_str(m.diag1_step)+"\n"
    nmlstr += " diag2_step = "+value_str(m.diag2_step)+"\n"
    nmlstr += " /"

    ##write the namelist to input.nml file
    with open(os.path.join(run_dir, 'input.nml'), 'wt') as f:
        f.write(nmlstr)

