import numpy as np
import matplotlib.pyplot as plt

def plot_obs_seq(obs_seq, obs_rec_id, v, vmin=-10, vmax=10, marker_size=20, cmap='bwr'):
    fig, ax = plt.subplots(1, 1, figsize=(4,4))

    obs = np.squeeze(obs_seq[obs_rec_id]['obs'][v, :])
    obs_x = obs_seq[obs_rec_id]['x']
    obs_y = obs_seq[obs_rec_id]['y']

    cmap = [plt.get_cmap(cmap)(x) for x in np.linspace(0, 1, round(vmax-vmin)+1)]
    cind = np.maximum(np.minimum(np.round(obs-vmin), int(np.round(vmax-vmin))), 0).astype(int)
    ax.scatter(obs_x, obs_y, marker_size, color=np.array(cmap)[cind, 0:3], edgecolor=None, linewidth=0.5)

