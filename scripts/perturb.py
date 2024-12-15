import os
import sys
import importlib.util
import numpy as np
from utils.conversion import dt1h, ensure_list
from utils.parallel import distribute_tasks, bcast_by_root, by_rank
from utils.progress import timer, print_with_cache, progress_bar
from utils.shell_utils import run_command, run_job
from utils.dir_def import forecast_dir, cycle_dir
from utils.random_perturb import random_perturb

def perturb(c):
    assert c.nproc==c.comm.Get_size(), f"Error: nproc {c.nproc} not equal to mpi size {c.comm.Get_size()}"

    ##clean perturb files in current cycle dir
    for rec in c.perturb:
        perturb_dir = os.path.join(forecast_dir(c, c.time, rec['model_src']), 'perturb')
        if c.pid==0:
            run_command(f"rm -rf {perturb_dir}; mkdir -p {perturb_dir}")
    c.comm.Barrier()

    ##distribute perturbation items among MPI ranks
    task_list = bcast_by_root(c.comm)(distribute_perturb_tasks)(c)

    c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)

    ##first go through the fields to count how many (for showing progress)
    nfld = 0
    for rec in task_list[c.pid]:
        model_name = rec['model_src']
        model = c.model_config[model_name]
        vname = ensure_list(rec['variable'])[0]
        dt = model.variables[vname]['dt']
        nstep = c.cycle_period // dt + 1
        for n in range(nstep):
            for k in model.variables[vname]['levels']:
                nfld += 1

    ##actually go through the fields to perturb now
    fld_id = 0
    for rec in task_list[c.pid]:
        model_name = rec['model_src']
        model = c.model_config[model_name]  ##model class object
        mem_id = rec['member']
        mstr = f'_mem{mem_id+1:03d}'
        path = forecast_dir(c, c.time, model_name)
        variable_list = ensure_list(rec['variable'])

        ##check if previous perturb is available from past cycles
        perturb = {}
        for vname in variable_list:
            psfile = os.path.join(forecast_dir(c, c.prev_time, model_name), 'perturb', vname+mstr+'.npy')
            if os.path.exists(psfile):
                perturb[vname] = np.load(psfile)
            else:
                perturb[vname] = None

        ##perturb all sub time steps for variables within this cycle
        dt = model.variables[vname]['dt']   ##time interval for variable vname in this cycle
        nstep = c.cycle_period // dt + 1
        for n in range(nstep):
            t = c.time + n * dt * dt1h

            #TODO: only works for surface layer variables now with k=0 (forcing variables)
            ##      but can be extended to 3D variables with additional vertical corr parameter
            for k in model.variables[vname]['levels']:
                fld_id += 1
                if c.debug:
                    print(f"PID {c.pid}: perturbing mem{mem_id+1} {variable_list} at {t} level {k}", flush=True)
                else:
                    print_1p(progress_bar(fld_id, nfld+1))

                vname =variable_list[0]  ##note: all variables in the list shall have same dt and k levels
                model.read_grid(path=path, name=vname, time=t, member=mem_id, k=k)
                model.grid.set_destination_grid(c.grid)
                c.grid.set_destination_grid(model.grid)

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
    print_1p(' done.\n\n')

def distribute_perturb_tasks(c):
    task_list_full = []
    for perturb_rec in ensure_list(c.perturb):
        for mem_id in range(c.nens):
            task_list_full.append({**perturb_rec, 'member':mem_id})
    task_list = distribute_tasks(c.comm, task_list_full)
    return task_list

def run(c):
    script_file = os.path.abspath(__file__)
    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    print(f"\033[1;33mRUNNING\033[0m {script_file}")
    if not hasattr(c, 'perturb') or c.perturb is None:
        print('no perturbation defined in config, exiting\n\n')
        return

    ##build run commands for the perturb script
    commands = f"source {c.python_env}; "

    if importlib.util.find_spec("mpi4py") is not None:
        commands += f"JOB_EXECUTE {sys.executable} -m mpi4py {script_file} -c {config_file}"
    else:
        print("Warning: mpi4py is not found, will try to run with nproc=1.", flush=True)
        commands += f"{sys.executable} {script_file} -c {config_file} --nproc=1"

    job_submit_opts = {}
    if c.job_submit:
        job_submit_opts = c.job_submit
        
    run_job(commands, job_name="perturb", run_dir=cycle_dir(c, c.time), nproc=c.nproc, **job_submit_opts)

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    print_1p = by_rank(c.comm, 0)(print_with_cache)
    print_1p("\nPerturbing the ensemble model state and forcing\n")
    if not hasattr(c, 'perturb') or c.perturb is None:
        print_1p('no perturbation defined in config, exiting\n\n')
        exit()

    timer(c)(perturb)(c)
