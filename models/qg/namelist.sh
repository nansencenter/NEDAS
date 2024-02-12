#!/bin/bash
. $config_file

forecast_steps=`echo "$forecast_period / 24 / $dt" |bc -l`

cat << EOF
&run_params
 kmax = $kmax
 nz = $nz

 F = ${F:-100}
 beta = ${beta:-20}

 adapt_dt = F
 dt = $dt

 psi_init_type = '${input_type:-read}'
 psi_init_file = 'input'
 initialize_energy = ${initialize_energy:-T}
 e_o = ${e_o:-1}
 k_o = ${k_o:-1}
 delk = ${delk:-3}
 m_o = ${m_o:-0}
 z_o = ${z_o:-0}

 strat_type = '${strat_type:-linear}'
 deltc = ${deltc:-0.1}
 ubar_type = '${ubar_type:-linear}'
 delu = ${delu:-0.1}
 uscale = ${uscale:-0.1}

 use_topo = ${use_topo:-F}
 topo_type = '${topo_type:-spectral}'
 k_o_topo = ${k_o_topo:-10}
 del_topo = ${del_topo:-3}

 use_forcing = ${use_forcing:-F}
 norm_forcing = ${norm_forcing:-F}
 forc_coef = ${forc_coef:-0}
 forc_corr = ${forc_corr:-0}
 kf_min = ${kf_min:-1}
 kf_max = ${kf_max:-3}

 filter_type = '${qg_filt_type:-exp_cutoff}'
 filter_exp = ${qg_filt_exp:-8}
 filt_tune = ${qg_filt_tune:-1}
 k_cut = ${k_cut:-45}

 bot_drag = ${bot_drag:-0.3}
 therm_drag = ${therm_drag:-0}
 rho_slope = ${rho_slope:-0.001}

 idum = ${random_seed:-0}

 total_counts = $forecast_steps
 write_step = $forecast_steps
 diag1_step = $forecast_steps
 diag2_step = $forecast_steps
 /
EOF
