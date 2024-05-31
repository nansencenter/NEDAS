##demo of how to run the qg model

from datetime import datetime

from models.qg import Model

model_param = {'kmax': 127,
               'psi_init_type': 'read'
               }

model = Model(**model_param)

run_opt = {'nedas_dir': '/cluster/home/yingyue/code/NEDAS',
           'host': 'betzy',
           'path': '/cluster/work/users/yingyue/qg/run',
           'time': datetime(2023,1,1),
           'input_file': '/cluster/work/users/yingyue/qg_ens_runs/0001/output.bin',
           'output_file': '/cluster/work/users/yingyue/qg/run/output.bin',
           }
model.run(**run_opt)

