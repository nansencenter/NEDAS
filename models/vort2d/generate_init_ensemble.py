import os
from config import Config
from utils.dir_def import forecast_dir

c = Config(parse_args=True)

model = c.model_config['vort2d']

ens_init_dir = model.ens_init_dir
os.system("mkdir -p "+ens_init_dir)

for m in range(c.nens):
    state = model.generate_initial_condition()

    model.write_var(state, path=ens_init_dir, name='velocity', is_vector=True, time=c.time, member=m)

