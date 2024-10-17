import os
from config import Config
from utils.conversion import dt1h
from utils.dir_def import forecast_dir

c = Config(parse_args=True)


model = c.model_config['vort2d']

truth_dir = model.truth_dir
os.system("mkdir -p "+truth_dir)

t = c.time_start
while t < c.time_end:

    if t == c.time_start:
        state = model.generate_initial_condition()
        model.write_var(state, path=truth_dir, name='velocity', is_vector=True, time=t)

    model.run(path=truth_dir, time=t, forecast_period=c.cycle_period)

    t += c.cycle_period * dt1h

