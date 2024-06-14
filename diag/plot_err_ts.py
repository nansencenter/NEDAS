import numpy as np
import sys
import matplotlib.pyplot as plt
from config import Config
from utils.conversion import s2t, t2s, dt1h

model_name = sys.argv[1] ##'qg'
c = Config(config_file="../config/samples/"+model_name+".yml")
c.nens = int(sys.argv[2])  ##5
c.work_dir = "/cluster/work/users/yingyue/nopert/"+model_name+f".n{c.nens}"

sc=''

c.time_start = s2t('202301010000')
c.time_end = s2t('20230218000')
t0 = c.time_start
t = t0
prev_t = t

h_ts = []
rmse_ts = []
sprd_ts = []
while t < c.time_end:
    next_t = t + dt1h * c.cycle_period
    h = (t - t0) / dt1h
    h_ts.append(h)
    path = c.work_dir+'/cycle/'+t2s(t)+'/'+model_name
    rmse_ts.append(np.load(path+"/rmse_prior"+sc+".npy"))
    sprd_ts.append(np.load(path+"/sprd_prior"+sc+".npy"))
    h_ts.append(h)
    rmse_ts.append(np.load(path+"/rmse_post"+sc+".npy"))
    sprd_ts.append(np.load(path+"/sprd_post"+sc+".npy"))

    prev_t = t
    t = next_t

fig, ax = plt.subplots(1, 1, figsize=(8,4))
ax.plot(h_ts, rmse_ts, 'b', linewidth=3, label='RMS error')
ax.plot(h_ts, sprd_ts, 'c', linewidth=1, label='spread')
ax.legend()
xticks = np.arange(0, (c.time_end-c.time_start)/dt1h+1, 72)
ax.set_xticks(xticks)
ax.set_xticklabels((xticks/24).astype(int))
ax.set_xlabel('time (day)')
ax.set_ylim(0, 1)
#ax.set_xlim(0, 120)
ax.set_title(model_name+r" $N_e$="+f"{c.nens} {sc}")

plt.savefig(f'sawtooth.{model_name}.n{c.nens}{sc}.png', dpi=100)

