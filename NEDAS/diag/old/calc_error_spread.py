import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from NEDAS.config import Config
from NEDAS.utils.conversion import t2s, s2t, dt1h

def get_scale(fld, krange, s):
    return fld
sc = ''

krange = (1, 5)
s = 0
# from utils.multiscale import get_scale
# sc = f"_s{s}"

nedas_dir = '/cluster/home/yingyue/code/NEDAS'
model_name = sys.argv[1] ##'qg'
c = Config(config_file=nedas_dir+"/config/samples/"+model_name+".yml")
c.nens = int(sys.argv[2]) ##20
c.work_dir = "/cluster/work/users/yingyue/"+model_name+f".n{c.nens}"

model = c.model_config[model_name]
c.time_start = s2t('202301010000')
c.time_end = s2t('20230218000')

varname='streamfunc'
vmin = -6
vmax = 6
is_vector = False

def calc_error_spread(t, prev_t):
    ##truth
    path = c.work_dir+'/truth/'+t2s(t)+'/'+model_name
    fld = model.read_var(path=path, name=varname, is_vector=is_vector, time=t)
    truth = get_scale(fld, krange, s)

    ##ens-mean prior
    path = c.work_dir+'/cycle/'+t2s(prev_t)+'/'+model_name
    ens_state_prior = np.full((c.nens,model.nz,model.ny,model.nx), np.nan)
    for m in range(c.nens):
        if not os.path.exists(model.filename(path=path, time=t, member=m)):
            continue
        fld = model.read_var(path=path, name=varname, is_vector=is_vector, time=t, member=m)
        ens_state_prior[m, ...] = get_scale(fld, krange, s)
    prior_mean = np.mean(ens_state_prior, axis=0)
    prior_var = np.sum((ens_state_prior-prior_mean)**2, axis=0) / (c.nens-1)

    ##ens-mean posterior
    path = c.work_dir+'/cycle/'+t2s(t)+'/'+model_name
    ens_state_post = np.full((c.nens,model.nz,model.ny,model.nx), np.nan)
    for m in range(c.nens):
        if not os.path.exists(model.filename(path=path, time=t, member=m)):
            continue
        fld = model.read_var(path=path, name=varname, is_vector=is_vector, time=t, member=m)
        ens_state_post[m, ...] = get_scale(fld, krange, s)
    post_mean = np.mean(ens_state_post, axis=0)
    post_var = np.sum((ens_state_post-post_mean)**2, axis=0) / (c.nens-1)

    rmse_prior = (np.sqrt(np.mean((prior_mean - truth)**2)))
    sprd_prior = (np.sqrt(np.mean(prior_var)))
    rmse_post = (np.sqrt(np.mean((post_mean - truth)**2)))
    sprd_post = (np.sqrt(np.mean(post_var)))

    path = c.work_dir+'/cycle/'+t2s(t)+'/'+model_name
    print(path)
    np.save(path+"/rmse_prior"+sc+".npy", rmse_prior)
    np.save(path+"/rmse_post"+sc+".npy", rmse_post)
    np.save(path+"/sprd_prior"+sc+".npy", sprd_prior)
    np.save(path+"/sprd_post"+sc+".npy", sprd_post)


t0 = c.time_start
t = t0
prev_t = t

with ProcessPoolExecutor(max_workers=100) as exe:
    futures = []
    while t < c.time_end:
        next_t = t + dt1h * c.cycle_period
        future = exe.submit(calc_error_spread, t, prev_t)
        futures.append(future)
        prev_t = t
        t = next_t
    for future in as_completed(futures):
        try:
            result = future.result()
        except Exception as e:
            print(f'error occurred: {e}')


