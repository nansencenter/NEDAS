import os
import sys
import subprocess

def makedir(dirname):
    try:
        os.makedirs(dirname, exist_ok=True)
    except FileExistsError:
        ##can happen if multiple processor are trying to make the same directory
        pass

def run_command(shell_cmd):
    """
    Run a shell command in a subprocess, handle errors
    """
    p = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0:
        raise RuntimeError(f"{p.stderr}")

def run_job(commands, job_name='run', nproc=1, offset=0, ppn=1,
            host='localhost', scheduler=None, project=None, queue=None, walltime=3600,
            run_separate_jobs=False, use_job_array=False, nens=1, **kwargs):
    """
    Top level job submission function, takes input args about a job and run it on a host machine.
    Inputs:
    -commands: list[str], linux command(s) to be run
    -job_name: str, name of the job
    -nproc: int, number of processors to use
    -offset: int, number of nodes to offset from the start
    -ppn: int, number of processors per compute node
    -host: str, host machine name
    -scheduler: str or None, type of scheduler used on the machine
    -project: str or None, project id for allocation
    -queue: str or None, which job queue to use
    -walltime: int, maximum run time allowed (in seconds)
    -run_separate_jobs: bool, if true, submit each job separately to the scheduler, otherwise run in same allocation.
    -use_job_array: bool, if true, use job array for ensemble members
    -nens: int, ensemble size
    """
    if run_separate_jobs:
        submit_job_and_monitor(commands, job_name, nproc, offset, ppn,
                               host, scheduler, project, queue, walltime, job_array_index)
    else:
        run_job_as_step(commands, nproc, offset, ppn, host, scheduler, walltime)

def get_resources(host, scheduler):
    if scheduler is None:
        res = {'nproc':1, 'nnode':1, 'ppn':1}
    else:
        ##get available resource counts
        res = {}
        if scheduler.lower() == 'slurm':
            res['nproc'] = int(os.environ['SLURM_NTASKS'])
            res['nnode'] = int(os.environ['SLURM_NNODES'])
            res['ppn'] = int(os.environ['SLURM_TASKS_PER_NODE'].split('(')[0])

        elif scheduler.lower() == 'oar':
            node_file = os.environ['OAR_NODE_FILE']
            # if node_file and os.path.exists(node_file)
            # with open(node_file, 'r') as file:
            #     node_list = [n for n in file]
            # ppn_avail = 32 ##TODO: how to get this from OAR environment variables? or is it okay to specify here?

        else:
            raise NotImplementedError(f"Scheduler type {scheduler} is not implemented in job_submit")

    ##some specific rules for host
    if host == 'betzy':
        r = subprocess.run("hostname", capture_output=True, text=True)
        if r.stdout.strip()[:5] == 'login':
            print("Warning: you are running job on Betzy login node, reducing to nproc=16")

    return res

def get_job_execute_command(res):
    if scheduler is None:
        execute_cmd = ''
    else:
        if scheduler.lower() == 'slurm':
            # execute_cmd = f"mpirun -np {nproc} -npernode {ppn} -machinefile $OAR_NODE_FILE"
            pass
    return execute_cmd

def make_job_submit_script(res):
    assert scheduler is not None
    if scheduler.lower() == 'slurm':
        ##header
        script = "#!/bin/bash\n"
        script += "\n"
        script += "\n"
        script += "\n"

        ##job_array_index
        ##execute command

    for command in commands:
        command = command.replace('JOB_ARRAY_INDEX', job_array_index_name)
        command = command.replace('JOB_EXECUTE', execute_cmd)

def submit_job_and_monitor(commands, job_name, nproc, offset, ppn,
                           host, scheduler, project, queue, walltime, job_array_index):
    """
    Run 'commands' by starting a separate job submission on 'host'
    Use 'nproc' processors and 'ppn' processors per compute node, limit to maximum 'walltime'
    In 'JOB_EXECUTE' in 'commands' will be replaced by the correct mpi executor for 'scheduler' on 'host'
    If job_array is true, 'JOB_ARRAY_INDEX' will be replaced by the scheduler's job array index
    """
    pass

def run_job_as_step(commands, nproc, offset, ppn, host, scheduler, walltime):
    """
    Run 'commands' from within a job allocation
    Use 'nproc' processors starting from the 'offset'+1 processors of the allocation
    In 'JOB_EXECUTE' in 'commands' will be replaced by the correct mpi executor for 'scheduler' on 'host'
    """
    nnode = nproc // ppn
    offset_node = offset // ppn
    # res = get_resources(host, scheduler)

    # assert nproc <= res['nproc'], f"Reuqested nproc={nproc} exceeds the available {res['nproc']} processors."
    # assert nnode+offset_node <= res['nnode'], f"Requested nnodes={nnode} and offset_node={offset_node} is out of bound of the available {res['nnode']} compute nodes."

    # execute_cmd = get_job_execute_command(res)

    # elif scheduler.lower() == 'oar':
    #     node_file_job = 'nodefile.tmp.$RAN'
    #     with open(node_file_job, 'wt') as file:
    #         file.write
    execute_cmd = "mpirun -np {nproc} ".format(nproc=nproc, offset=offset_node)
    commands = commands.replace('JOB_EXECUTE', execute_cmd)

    p = subprocess.Popen(commands, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
    p.wait()

    ##handle error
    if p.returncode != 0:
        print(f"{script_path} raised error: {p.stderr}")
        sys.exit(1)

