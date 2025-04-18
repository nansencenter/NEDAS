import numpy as np
import matplotlib.pyplot as plt
from config import Config
from utils.conversion import s2t, t2s, dt1h

model_list = ['qg', 'qg.emulator']
colors = ['k', 'r']
cost_factor = [1, 0.1]
nens_list = [[5, 10, 20, 40, 60, 120, 240],
             [40, 60, 120, 240, 480, 960]]

fig, ax = plt.subplots(1, 1, figsize=(6,6))

for i in range(len(model_list)):
    model_name = model_list[i]
    err_means = []
    for j in range(len(nens_list[i])):
        nens = nens_list[i][j]
        print(model_name, nens)

        work_dir = "/cluster/work/users/yingyue/nopert/"+model_name+f".n{nens}"
        sc=''

        ##collect err from time series
        time_start = s2t('202301011200')
        time_end = s2t('20230105000')
        cycle_period = 12
        t0 = time_start
        t = t0
        prev_t = t
        err = []
        while t < time_end:
            next_t = t + dt1h * cycle_period
            path = work_dir+'/cycle/'+t2s(t)+'/'+model_name
            err.append(np.load(path+"/rmse_post"+sc+".npy"))
            prev_t = t
            t = next_t

        err_mean = np.mean(err)
        err_means.append(err_mean)

        ##plot the err std as whiskers
        err_std = np.std(err)
        ax.semilogx(cost_factor[i]*np.array([nens, nens]), np.array([err_mean-err_std, err_mean+err_std]), color=colors[i])

    ##plot the mean error as a curve
    ax.semilogx(cost_factor[i]*np.array(nens_list[i]), np.array(err_means), marker='o', color=colors[i], label=model_name)

ax.grid(axis='y')
ax.legend(fontsize=15)
ax.set_xticks([5, 10, 20, 40, 60, 120, 240])
ax.set_xticklabels([5, 10, 20, 40, 60, 120, 240])
ax.set_xlabel(r'$N_e$', fontsize=12)
ax.set_title('Analysis error over first 10 cycles', fontsize=15)

plt.savefig(f'err_summary.png', dpi=100)

