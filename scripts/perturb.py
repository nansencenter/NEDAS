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

def get_prev_perturb(c, rec):
#     pass

def perturb_field():
    pass

def main(c):
    task_list = bcast_by_root(c.comm)(distribute_perturb_tasks)(c)

    for rec in task_list[c.pid]:
        model_name = rec['model_src']
        model = c.model_config[model_name]
        vname = rec['variable']
        mem_id = rec['member']
        path = forecast_dir(c, c.time, model_name)

        t0 = t2h(c.time)
        dt = model.variables[vname]['dt']
        nstep = c.cycle_period / dt
        for d in np.arange(c.cycle_period/dt):
            # print(f"perturbing {model_name} {rec['variable']}", flush=True)

            ##read variable from model state
            fld = model.read_var(path=path, name=vname, time=h2t(t), member=mem_id)

            prev_pert_file = os.path.join(path, 'perturb.'+vname+'.npy')
            if os.path.exists(prev_pert_file):
                prev_perturb = np.load(prev_pert_file)
            else:
                prev_perturb = None

            ##generate perturbation
            fld_new, pert = random_perturb(model.grid, fld, prev_perturb, **rec)

            ##save a copy of perturbation for later cycle use
            np.save(prev_pert_file, pert)

            ##write variable back to model state
            model.write_var(fld_new, path=path, name=vname, time=h2t(t), member=mem_id)


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

