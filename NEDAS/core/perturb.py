from .context import Context

class Perturbation:

    def __init__(self, ):
        pass

    def __call__(self, c: Context):

    def perturb(self, c):

        if c.perturb is None:
            c.print_1p(f"No perturbation defined in config, exiting.\n")
            return
        c.print_1p(f"Perturbing state:")

        ##clean perturb files in current cycle dir
        for rec in c.perturb:
            perturb_dir = os.path.join(c.forecast_dir(c.time, rec['model_src']), 'perturb')
            if c.pid==0:
                run_command(f"rm -rf {perturb_dir}; mkdir -p {perturb_dir}")
        c.comm.Barrier()

        ##distribute perturbation items among MPI ranks
        task_list = bcast_by_root(c.comm)(self.distribute_perturb_tasks)(c)

        c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]

        ##first go through the fields to count how many (for showing progress)
        nfld = 0
        for rec in task_list[c.pid]:
            model_name = rec['model_src']
            model = c.models[model_name]
            vname = ensure_list(rec['variable'])[0]
            dt = model.variables[vname]['dt']
            niter = c.cycle_period // dt + 1
            for n in range(niter):
                for k in model.variables[vname]['levels']:
                    nfld += 1

        ##actually go through the fields to perturb now
        fld_id = 0
        for rec in task_list[c.pid]:
            model_name = rec['model_src']
            model = c.models[model_name]  ##model class object
            mem_id = rec['member']
            mstr = f'_mem{mem_id+1:03d}'
            path = c.forecast_dir(c.time, model_name)
            variable_list = ensure_list(rec['variable'])

            ##check if previous perturb is available from past cycles
            perturb = {}
            for vname in variable_list:
                psfile = os.path.join(c.forecast_dir(c.prev_time, model_name), 'perturb', vname+mstr+'.npy')
                if os.path.exists(psfile):
                    perturb[vname] = np.load(psfile)
                else:
                    perturb[vname] = None

            # get number of time steps for this set of variables
            # perturbation will be generated for all time steps if variable is available
            dt = max([model.variables[v]['dt'] for v in variable_list])
            nstep = c.cycle_period // dt + 1
            for n in range(nstep):
                t = c.time + n * dt * dt1h

                # TODO: perturbation for each k level is drawn independently, can be improved
                # by introducing a vertical correlation length scale, or using EOF modes.
                # Note: assuming all variables in the list have the same k levels
                for k in model.variables[variable_list[0]]['levels']:
                    fld_id += 1
                    if c.debug:
                        print(f"PID {c.pid:4}: perturbing mem{mem_id+1:03} {variable_list} at {t} level {k}", flush=True)
                    else:
                        c.print_1p(progress_bar(fld_id, nfld+1))

                    vname =variable_list[0]  ##note: all variables in the list shall have same dt and k levels
                    model.read_grid(path=path, name=vname, time=t, member=mem_id, k=k)
                    model.grid.set_destination_grid(c.grid)
                    c.grid.set_destination_grid(model.grid)

                    # collect variable fields
                    fields = {}
                    for vname in variable_list:
                        ##read variable from model state
                        fld = model.read_var(path=path, name=vname, time=t, member=mem_id, k=k)
                        ##convert to analysis grid
                        fields[vname] = model.grid.convert(fld, is_vector=model.variables[vname]['is_vector'])

                    ##generate perturbation on analysis grid
                    fields_pert, perturb = random_perturb(c.grid, fields, prev_perturb=perturb, dt=dt, n=n, **rec)

                    if rec['type'].split(',')[0]=='displace' and hasattr(model, 'displace'):
                        ##use model internal method to apply displacement perturbations directly
                        model.displace(perturb, path=path, time=t, member=mem_id, k=k)
                    else:
                        ##convert from analysis grid to model grid, and
                        ##write the perturbed variables back to model state files
                        for vname in variable_list:
                            fld = c.grid.convert(fields_pert[vname], is_vector=model.variables[vname]['is_vector'])
                            model.write_var(fld, path=path, name=vname, time=t, member=mem_id, k=k)

            ##save a copy of perturbation at next_t, for use by next cycle
            for vname in variable_list:
                psfile = os.path.join(path, 'perturb', vname+mstr+'.npy')
                run_command(f"mkdir -p {os.path.dirname(psfile)}")
                np.save(psfile, perturb[vname])

        c.comm.Barrier()
        c.print_1p(' done.\n')

    def distribute_perturb_tasks(self, c):
        task_list_full = []
        for perturb_rec in ensure_list(c.perturb):
            for mem_id in range(c.nens):
                task_list_full.append({**perturb_rec, 'member':mem_id})
        task_list = distribute_tasks(c.comm, task_list_full)
        return task_list



def random_perturb(grid, fields, prev_perturb, dt=1, n=0, seed=None, **kwargs):
    """
    Add random perturbation to the given 2D field

    Args:
        grid: Grid object describing the 2d domain
        fields: list of np.array shape[...,ny,nx]
        prev_perturb; list of np.array from previous perturbation data, None if unavailable
        dt: float, interval (hours) between time steps
        n: int, current time step index
        variable: str, or list of str
        type: str: 'gaussian', 'powerlaw', or 'displace'
        amplitude: float, (or list of floats, in multiscale approach)
        hcorr: float, or list of float, horizontal corr length (meters)
        tcorr: float, or list of float, time corr length (hours)
        **kwargs: other arguments
    """
    if seed is None:
        ##try to randomize using system entropy
        seed = int.from_bytes(os.urandom(4), 'little')
    else:
        assert isinstance(seed, int)
    np.random.seed(seed)

    perturb = {}
    perturb_type, other_opts, params = parse_perturb_opts(**kwargs)

    for vname,rec in params.items():
        fld = fields[vname]
        assert grid.x.shape == fld.shape[-2:], f"input fields[{vname}] dimension mismatch with grid"

        ns = rec['nscale']
        if perturb_type == 'displace':
            perturb[vname] = np.zeros((ns,2)+fld.shape[-2:])
        else:
            perturb[vname] = np.zeros((ns,)+fld.shape)

        if prev_perturb[vname] is not None and n==0:
            perturb[vname] = prev_perturb[vname]
            continue

        ##loop over scale s and generate perturbation
        for s in range(ns):
            ##draw a random field for each 2d field component in fields
            for ind in np.ndindex(fld.shape[:-2]):
                if perturb_type == 'gaussian':
                    perturb[vname][(s,)+ind] = random_field_gaussian(grid.nx, grid.ny, rec['amp'][s], rec['hcorr'][s]/grid.dx)

                elif perturb_type == 'powerlaw':
                    perturb[vname][(s,)+ind] = random_field_powerlaw(grid.nx, grid.ny, rec['amp'][s], rec['powerlaw'][s])

                elif perturb_type == 'displace':
                    mask = np.full(grid.x.shape, False)
                    du, dv = random_displacement(grid, mask, rec['amp'][s], rec['hcorr'][s]/grid.dx)
                    perturb[vname][s] = np.array([du, dv])

                else:
                    raise NotImplementedError('unknown perturbation type: '+perturb_type)

            ##create perturbations that are correlated in time
            autocorr = 0.75
            ncorr = rec['tcorr'][s] / dt  ##time steps at decorrelation
            alpha = autocorr**(1.0 / ncorr)
            if prev_perturb[vname] is not None:
                perturb[vname][s] = np.sqrt(1-alpha**2) * perturb[vname][s] + alpha * prev_perturb[vname][s]

    ###legacy prsflg==1,2 options in force_perturb program, reproduced here
    if 'press_wind_relate' in other_opts:
        for vname in ['atmos_surf_velocity', 'atmos_surf_press']:
            assert vname in params.keys(), f'{vname} not in variable list, cannot run press_wind_relate option'

        for s in range(params['atmos_surf_press']['nscale']):
            perturb['atmos_surf_velocity'][s] = get_velocity_from_press(grid, perturb['atmos_surf_press'][s], ('scale_wind' in other_opts), params['atmos_surf_press']['amp'][s], params['atmos_surf_press']['hcorr'][s], params['atmos_surf_velocity']['amp'][s])

    ##now add perturbations to each field
    for vname,rec in params.items():
        for s in range(rec['nscale']):
            if perturb_type == 'displace':
                fields[vname] = warp(grid, fields[vname], perturb[vname][s,0,...], perturb[vname][s,1,...])

            else:
                if 'exp' in other_opts:
                    ##add lognormal perturbations
                    fields[vname] *= np.exp(perturb[vname][s,...] - 0.5*rec['amp'][s]**4)

                else:
                    ##just add the gaussian perturbations
                    fields[vname] += perturb[vname][s,...]

        ##respect value bounds after perturbing
        if 'bounds' in kwargs:
            vmin, vmax = kwargs['bounds']
            fields[vname] = np.minimum(np.maximum(fields[vname], vmin), vmax)

    return fields, perturb

def parse_perturb_opts(**kwargs):
    ##perturb['type'] string format:
    #main option (gaussian/powerlaw/displace) followed by , then additional options separated by ,
    opts = kwargs['type'].split(',')
    perturb_type = opts[0]
    other_opts = []
    for opt in opts[1:]:
        other_opts.append(opt)

    key_list = []
    for key in ['amp', 'hcorr', 'tcorr', 'powerlaw']:
        if key in kwargs:
            key_list.append(key)

    ##a list of variables can be specified if running a multivariate perturbation scheme
    ##rectify variable and parameter to be lists for further processing
    if not isinstance(kwargs['variable'], list):
        kwargs['variable'] = [kwargs['variable']]
        for key in key_list:
            kwargs[key] = [kwargs[key]]
    variable_list = kwargs['variable']
    nv = len(variable_list)  ##number of variables

    ##ensure again that parameters are rectified to lists 
    for key in key_list:
        if not isinstance(kwargs[key], list):
            kwargs[key] = [kwargs[key]]

        ##check for mismatch in list length    
        if len(kwargs[key]) != nv:
            raise ValueError(f"perturb option: {key} has {len(kwargs[key])} entries, but {nv} variables are specified")

    ##get perturbation parameters for each variable from kwargs
    params = {}
    for v in range(nv):
        vname = variable_list[v]
        params[vname] = {}
        ##in multiscale approach, a list of parameters can be specified for a variable;
        ##one separate perturbation will be generated for each, then they will be added together
        if isinstance(kwargs[key_list[0]][v], list):
            nscale = len(kwargs[key_list[0]][v])
        else:
            nscale = 1
            for key in key_list:  ##make a list even if only one value for the key
                kwargs[key][v] = [kwargs[key][v]]
        params[vname]['nscale'] = nscale
        ##check if all keys are lists with same len
        for key in key_list[1:]:
            assert len(kwargs[key][v]) == nscale, f"perturb option: {key} has different number of entries from {key_list[0]}, check config"
        ##assign the parameters
        for key in key_list:
            params[vname][key] = kwargs[key][v]

    return perturb_type, other_opts, params


