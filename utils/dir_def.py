import os
from utils.conversion import t2s

def cycle_dir(c, time):
    """directory name for cycle at time
    Inputs: c: config object; time: datetime object
    Return: dir name string
    """
    return os.path.join(c.work_dir, 'cycle', t2s(time))

def forecast_dir(c, time, model_name):
    return os.path.join(cycle_dir(c, time), model_name)

def analysis_dir(c, time):
    ##multi scale components
    if c.nscale == 1:
        scale_dir = ''
    else:
        scale_dir = f'scale{c.scale_id}'
    return os.path.join(cycle_dir(c, time), 'analysis', scale_dir)

