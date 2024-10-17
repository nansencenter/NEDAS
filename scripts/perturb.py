import numpy as np
import os
from utils.conversion import dt1h, ensure_list
from utils.parallel import distribute_tasks, bcast_by_root, by_rank
from utils.progress import timer, print_with_cache, progress_bar
from utils.dir_def import forecast_dir
from utils.random_perturb import random_perturb

perturb_script_path = os.path.abspath(__file__)

def perturb(c):
    ##clean perturb files in current cycle dir
    for rec in c.perturb:
        perturb_files = os.path.join(forecast_dir(c, c.time, rec['model_src']), 'perturb', '*')
        if c.pid==0:
            os.system(f"rm -f {perturb_files}")
    c.comm.Barrier()

    ##distribute perturbation items among MPI ranks
    task_list = bcast_by_root(c.comm)(distribute_perturb_tasks)(c)

    c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)
    print_1p("\nPerturbing the ensemble model state and forcing\n")

    ##first go through the fields to count how many (for showing progress)
    nfld = 0
    for rec in task_list[c.pid]:
        model_name = rec['model_src']
        model = c.model_config[model_name]
        vname = ensure_list(rec['variable'])[0]
        dt = model.variables[vname]['dt']
        nstep = c.cycle_period // dt
        for n in np.arange(nstep):
            for k in model.variables[vname]['levels']:
                nfld += 1

    ##actually go through the fields to perturb now
    fld_id = 0
    for rec in task_list[c.pid]:
        model_name = rec['model_src']
        model = c.model_config[model_name]  ##model class object
        mem_id = rec['member']
        path = forecast_dir(c, c.time, model_name)
        variable_list = ensure_list(rec['variable'])

        ##check if previous perturb is available from past cycles
        perturb = {}
        for vname in variable_list:
            psfile = perturb_save_file(c, model_name, vname, c.prev_time, mem_id)
            if os.path.exists(psfile):
                perturb[vname] = np.load(psfile)
            else:
                perturb[vname] = None

        ##perturb all sub time steps for variables within this cycle
        dt = model.variables[vname]['dt']   ##time interval for variable vname in this cycle
        nstep = c.cycle_period // dt        ##number of time steps to be perturbed
        for n in np.arange(nstep):
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
                ##inside random_perturb: figure out grid convertion
                fields_pert, perturb = random_perturb(c.grid, fields, prev_perturb=perturb, dt=dt, **rec)

                if rec['type']=='displace' and hasattr(model, 'displace'):
                    ##use model internal method to apply displacement perturbations directly
                    model.displace(perturb, path=path, time=t, member=mem_id, k=k)
                else:
                    ##convert from analysis grid to model grid, and
                    ##write the perturbed variables back to model state files
                    for vname in variable_list:
                        fld = c.grid.convert(fields_pert[vname], is_vector=model.variables[vname]['is_vector'])
                        model.write_var(fld, path=path, name=vname, time=t, member=mem_id, k=k)

        ##save a copy of perturbation for later cycles
        for vname in variable_list:
            psfile = perturb_save_file(c, model_name, vname, c.time, mem_id)
            os.system(f"mkdir -p {os.path.dirname(psfile)}")
            np.save(psfile, perturb[vname])
    c.comm.Barrier()
    print_1p(' done.\n')

def distribute_perturb_tasks(c):
    task_list_full = []
    for perturb_rec in ensure_list(c.perturb):
        for mem_id in range(c.nens):
            task_list_full.append({**perturb_rec, 'member':mem_id})
    task_list = distribute_tasks(c.comm, task_list_full)
    return task_list

def perturb_save_file(c, model_name, vname, time, mem_id):
    path = forecast_dir(c, time, model_name)
    mstr = f'_mem{mem_id+1:03d}'
    return os.path.join(path, 'perturb', vname+mstr+'.npy')

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    if not hasattr(c, 'perturb') or c.perturb is None:
        exit()

    timer(c)(perturb)(c)

