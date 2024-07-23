##demo of how to run the qg model

from datetime import datetime

from models.qg import Model

model_param = {'kmax': 127,
               'psi_init_type': 'read'
               }

model = Model(**model_param)

run_opt = {'nedas_dir': '/cluster/home/yingyue/code/NEDAS',
           'job_submit_cmd': '/cluster/home/yingyue/code/NEDAS/config/samples/job_submit_betzy.sh',
           'model_code_dir': '/cluster/home/yingyue/code/NEDAS/models/qg',
           'model_data_dir': '',
           'path': '/cluster/work/users/yingyue/qg/run',
           'time': datetime(2023,1,1),
           'forecast_period': 12,
           }
model.run(**run_opt)

