import numpy as np
import os
from utils.conversion import dt1h, ensure_list
from utils.parallel import distribute_tasks, bcast_by_root, by_rank
from utils.progress import timer, print_with_cache, progress_bar
from utils.dir_def import forecast_dir, analysis_dir

def distribute_diag_tasks(c):
    task_list_full = []
    for diag_rec in c.diag:
        for mem_id in range(c.nens):
            task_list_full.append({**diag_rec, 'member':mem_id})
    task_list = distribute_tasks(c.comm, task_list_full)
    return task_list

def main(c):
    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    if c.debug:
        print('perturbing \n')

    task_list = bcast_by_root(c.comm)(distribute_perturb_tasks)(c)
    nr = len(task_list[c.pid])
    for r in range(nr):
        rec = task_list[c.pid][r]
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

        dt = model.variables[vname]['dt']   ##time interval for variable vname in this cycle
        nstep = c.cycle_period // dt        ##number of time steps to be perturbed
        for n in np.arange(nstep):
            if c.debug:
                print(progress_bar(r*nstep+n, nr*nstep))

            t = c.time + n * dt * dt1h

            for k in model.variables[vname]['levels']:
                ##TODO: only works for surface layer variables now (forcing)
                ##      but can be extended to 3D variables with additional vertical corr

                ##read variable from model state
                fields = {}
                for vname in variable_list:
                    fields[vname] = model.read_var(path=path, name=vname, time=t, member=mem_id, k=k)

                ##generate perturbation
                fields_pert, perturb = random_perturb(model.grid, fields, prev_perturb=perturb, dt=dt, **rec)

                ##write variable back to model state
                for vname in variable_list:
                    model.write_var(fields_pert[vname], path=path, name=vname, time=t, member=mem_id, k=k)

        ##save a copy of perturbation for later cycles
        for vname in variable_list:
            psfile = perturb_save_file(c, model_name, vname, c.time, mem_id)
            os.system(f"mkdir -p {os.path.dirname(psfile)}")
            np.save(psfile, perturb[vname])

    if c.debug:
        print('. done\n')

def diag(c):
    """run this diag.py script in a subprocess with mpi, using all nproc cores"""

    config_file = os.path.join(c.work_dir, 'config.yml')
    c.dump_yaml(config_file)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    diag_code = os.path.join(script_dir, 'diag.py')

    shell_cmd = c.job_submit_cmd+f" {c.nproc} {0}"
    shell_cmd += " python -m mpi4py "+diag_code
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
