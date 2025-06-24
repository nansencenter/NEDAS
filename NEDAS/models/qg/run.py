##demo of how to run the qg model in standalone mode
from datetime import datetime
from NEDAS.models.qg import Model

def main():
    ##run the model by
    ##python models/qg/run.py -c models/qg/default.yml

    ##model config variables are listed in default.yml
    ##settings can also be changed in model_param
    model_param = {
        'kmax': 127,
        'psi_init_type': 'spectral_m',
        'initialize_energy': True,
        'model_env': '/cluster/home/yingyue/code/NEDAS/models/qg/env/setup_betzy.src',
        'model_code_dir': '/cluster/home/yingyue/code/NEDAS/models/qg',
        }

    run_opt = {
        'job_submit_cmd': '/cluster/home/yingyue/code/NEDAS/config/samples/job_submit_betzy.sh',
        'path': '/cluster/work/users/yingyue/qg_run',
        'time': datetime(2023,1,1),
        'forecast_period': 120,
        }

    model = Model(parse_args=True, **model_param)

    model.run(**run_opt)

if __name__ == '__main__':
    main()

