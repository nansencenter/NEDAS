from datetime import datetime as dt

from models.qg import Model


m = Model()
m.run('/cluster/work/users/yingyue/qg/run', time=dt(2001,1,1), member=0)

# def run(c, m, **kwargs):
#     # m.nedas_dir = '/cluster/home/yingyue/code/NEDAS'
#     # m.host = 'betzy'
#     # m.work_dir = '/cluster/work/users/yingyue/qg'
#     # m.time = s2t('202301010000')

#     ##setup run directory
#     if hasattr(c, 'work_dir'):
#         run_dir = os.path.join(c.work_dir, 'cycle', t2s(c.time), 'qg')
#     else:
#         run_dir = '.'

#     if 'member' in kwargs:
#         run_dir = os.path.join(run_dir, '{:04d}'.format(kwargs['member']+1))

#     print('running model in '+run_dir)
#     if not os.path.exists(run_dir):
#         os.makedirs(run_dir)

#     os.chdir(run_dir)

#     ##setup shell environment
#     # if not hasattr(c, 'host'):
#     #     if 'host' in kwargs:
#     #         c.host = kwargs['host']
#     #     else:

#     # else:

#     ##parse model config file
#     # if 'config_file' in kwargs:
#     #     config_file = kwargs['config_file']
#     # else:
#     #     config_file = None
#     # model_config = parse_config(config_file)

#     namelist(m)

#     ##clean up before run
#     subprocess.run('rm -f restart.nml *bin', shell=True)

#     ##link the input file


#     ##build the shell command line
#     env_dir = os.path.join(c.nedas_dir, 'config', 'env', c.host)
#     # submit_cmd = os.path.join(env_dir, 'job_submit.sh')+f" {nnode} {nproc} {offset} "
#     submit_cmd = ''
#     qg_exe_path = os.path.join(c.nedas_dir, 'models', 'qg', 'src', 'qg.exe')

#     shell_cmd = "source "+os.path.join(env_dir, 'qg.src')+"; "
#     shell_cmd += submit_cmd
#     shell_cmd += qg_exe_path+" . "
#     shell_cmd += ">& run.log"

#     # print(shell_cmd)
#     p = subprocess.Popen(shell_cmd, shell=True)

#     # Check the status of the process
#     elapsed_time = 0
#     check_time_step = 1
#     timeout = 100
#     while True:
#         # Use poll() to check if the process has terminated
#         status = p.poll()
#         if status is None:
#             elapsed_time += check_time_step
#         else:
#             print(f"Process terminated with exit code {status}")
#             break
#         time.sleep(check_time_step)
#         if elapsed_time > timeout:
#             print("taking too long, killing")
#             p.kill()
#             break

#     ##collect output


# if __name__ == '__main__':
#     from config import Config
#     from models.qg.model import Model
#     c = Config()
#     m = Model(parse_args=True)
#     run(c,m)

