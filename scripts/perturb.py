import numpy as np
import os
from utils.conversion import dt1h
from utils.parallel import distribute_tasks, bcast_by_root
from utils.progress import timer
from utils.dir_def import forecast_dir
from utils.random_perturb import random_perturb

def distribute_perturb_tasks(c):
    task_list_full = []
    for perturb_rec in c.perturb:
        for mem_id in range(c.nens):
            task_list_full.append({**perturb_rec, 'member':mem_id})
    task_list = distribute_tasks(c.comm, task_list_full)
    return task_list

def perturb_save_file(c, model_name, vname, time, mem_id):
    path = forecast_dir(c, time, model_name)
    mstr = f'_mem{mem_id+1:03d}'
    return os.path.join(path, 'perturb', vname+mstr+'.npy')

def main(c):
    task_list = bcast_by_root(c.comm)(distribute_perturb_tasks)(c)

    for rec in task_list[c.pid]:
        model_name = rec['model_src']
        model = c.model_config[model_name]  ##model class object
        vname = rec['variable']
        mem_id = rec['member']
        path = forecast_dir(c, c.time, model_name)

        ##check if previous perturb is available from past cycles
        psfile = perturb_save_file(c, model_name, vname, c.prev_time, mem_id)
        if os.path.exists(psfile):
            perturb = np.load(psfile)
        else:
            perturb = None

        dt = model.variables[vname]['dt']   ##time interval for variable vname in this cycle
        nstep = c.cycle_period // dt        ##number of time steps to be perturbed
        for n in np.arange(nstep):
            t = c.time + n * dt * dt1h

            for k in model.variables[vname]['levels']:
                ##TODO: only works for surface layer variables now (forcing)
                ##      but can be extended to 3D variables with additional vertical corr

                ##read variable from model state
                fld = model.read_var(path=path, name=vname, time=t, member=mem_id, k=k)

                ##generate perturbation
                fld_pert, perturb = random_perturb(model.grid, fld, prev_perturb=perturb, dt=dt, **rec)

                ##write variable back to model state
                model.write_var(fld_pert, path=path, name=vname, time=t, member=mem_id, k=k)

        ##save a copy of perturbation for later cycles
        psfile = perturb_save_file(c, model_name, vname, c.time, mem_id)
        os.system(f"mkdir -p {os.path.dirname(psfile)}")
        np.save(psfile, perturb)

def perturb(c):
    """run this perturb.py script in a subprocess with mpi, using all nproc cores"""

    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    perturb_code = os.path.join(script_dir, 'perturb.py')

    shell_cmd = c.job_submit_cmd+f" {c.nproc} {0}"
    shell_cmd += " python -m mpi4py "+perturb_code
    shell_cmd += " --config_file "+config_file

    p = subprocess.Popen(shell_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)

    p.wait()
    if p.returncode != 0:
        print(f"{p.stderr}")
        exit()

if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    main(c)

