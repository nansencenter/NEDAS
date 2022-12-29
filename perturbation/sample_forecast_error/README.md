specify perturbation parameters: vars, hradius, tradius as a function of spatial scale and time index

the parameters can be varying over the period (errors grow in time and saturate during the forecast). scale dependency: different variance and correlation length scales are associated with different scale bands, these parameters can either be manually specified or derived from samples of real forecast errors.

directory: $path_to_perturb_dir/param/$time/$variable/{vars,hradius,tradius}
format: text file with float number.
