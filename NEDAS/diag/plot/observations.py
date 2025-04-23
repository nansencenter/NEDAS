"""Diagnostic module to plot the observation sequence and obs_prior ensemble"""

import os
import numpy as np
import matplotlib.pyplot as plt
from NEDAS.utils.conversion import ensure_list, dt1h
from NEDAS.utils.shell_utils import makedir
from NEDAS.utils.graphics import add_colorbar, adjust_ax_size, get_cmap
from NEDAS.assim_tools.state import State
from NEDAS.assim_tools.obs import Obs

def get_task_list(c, **kwargs) -> list:

    variables = ensure_list(kwargs['variables'])
    dataset_src = ensure_list(kwargs['dataset_src'])
    vmin = ensure_list(kwargs['vmin'])
    vmax = ensure_list(kwargs['vmax'])
    nlevels = ensure_list(kwargs['nlevels'])
    cmap = ensure_list(kwargs['cmap'])
    vmin_diff = ensure_list(kwargs['vmin_diff'])
    vmax_diff = ensure_list(kwargs['vmax_diff'])
    nlevels_diff = ensure_list(kwargs['nlevels_diff'])
    cmap_diff = ensure_list(kwargs['cmap_diff'])

    state = State(c)
    obs = Obs(c, state)

    ##observation time steps within window
    obs_window_min = kwargs.get('obs_window_min', 0)
    obs_window_max = kwargs.get('obs_window_max', 0)
    obs_dt = ensure_list(kwargs['obs_dt'])
    obs_kmin = ensure_list(kwargs['obs_kmin'])
    obs_kmax = ensure_list(kwargs['obs_kmax'])

    tasks = []
    for i, vname in enumerate(variables):
        ##check if obs rec is defined in obs.info
        obs_rec_query = [id for id,r in obs.info['records'].items() if r['name']==vname and r['dataset_src']==dataset_src[i]]
        assert len(obs_rec_query)>0, f"cannot find obs record for '{vname}' from dataset '{dataset_src[i]}'"
        obs_rec_id = obs_rec_query[0]

        ##time steps for this obs
        obs_ts = c.time + np.arange(obs_window_min, obs_window_max, obs_dt[i]) * dt1h

        ##vertical levels for this obs
        levels = np.arange(obs_kmin[i], obs_kmax[i]+1)

        for k in levels:
            for t in obs_ts:
                for m in range(c.nens):
                    tasks.append({**kwargs, 'obs_rec_id':obs_rec_id, 'member':m, 'k':k, 't':t, 'dt':obs_dt[i], 'vmin':vmin[i], 'vmax':vmax[i], 'nlevels':nlevels[i], 'cmap':cmap[i], 'vmin_diff':vmin_diff[i], 'vmax_diff':vmax_diff[i], 'nlevels_diff':nlevels_diff[i], 'cmap_diff':cmap_diff[i]})
    return tasks

def run(c, **kwargs) -> None:
    """
    Run diagnostics: plot the ensemble states
    """
    if 'plot_dir' in kwargs:
        plot_dir = kwargs['plot_dir']
    else:
        plot_dir = os.path.join(c.work_dir, 'plots', 'observations')
    makedir(plot_dir)

    state = State(c)
    obs = Obs(c, state)

    figsize = (kwargs.get('fig_size_x', 16), kwargs.get('fig_size_y', 7))
    landcolor = kwargs.get('land_color', 'gray')

    obs_rec_id = kwargs['obs_rec_id']
    member = kwargs['member']
    obs_rec = obs.info['records'][obs_rec_id]
    vmin = kwargs['vmin']
    vmax = kwargs['vmax']
    nlevels = kwargs['nlevels']
    cmap = get_cmap(kwargs['cmap'])
    vmin_diff = kwargs['vmin_diff']
    vmax_diff = kwargs['vmax_diff']
    nlevels_diff = kwargs['nlevels_diff']
    cmap_diff = get_cmap(kwargs['cmap_diff'])

    k = kwargs['k']
    t = kwargs['t']
    dt = kwargs['dt']
    if c.debug:
        print(f"PID {c.pid:4} plotting observations '{obs_rec['name']:20}' from {obs_rec['dataset_src']} at level {k:3} {t} ~ {t+dt*dt1h}", flush=True)

    ##if the viewer html file does not exist, generate it
    viewer = os.path.join(plot_dir, 'index.html')
    if not os.path.exists(viewer):
        generate_viewer_html(c, plot_dir, figsize, **kwargs)

    ##plot the variables defined in kwargs, save to figfile
    figfile = os.path.join(plot_dir, f"{obs_rec['dataset_src']}_{obs_rec['name']}_k{k}_{t:%Y%m%dT%H%M%S}_{t+dt*dt1h:%Y%m%dT%H%M%S}_mem{member+1:03}.png")

    ##read the obs data from analysis_dir/obs_seq
    obs_seq = np.load(os.path.join(state.analysis_dir, f'obs_seq.rec{obs_rec_id}.npy'), allow_pickle=True).item()
    obs_prior_seq = np.load(os.path.join(state.analysis_dir, f'obs_prior_seq.rec{obs_rec_id}.mem{member:03}.npy'), allow_pickle=True)

    ##filter for the obs within time and vertical level range
    tmask = (obs_seq['t'] > t) & (obs_seq['t'] <= t+dt*dt1h)
    obs_z = np.abs(obs_seq['z'])
    obs_x = obs_seq['x']
    obs_y = obs_seq['y']
    model_z = np.abs(obs.read_mean_z_coords(c, state, c.time))
    if k == 0:
        zk = c.grid.interp(model_z[k], obs_x, obs_y)
        zmask = (obs_seq['z'] == zk)
    else:
        zk = c.grid.interp(model_z[k], obs_x, obs_y)
        zk1 = c.grid.interp(model_z[k-1], obs_x, obs_y)
        zmask = (obs_z > zk1) & (obs_z <= zk)
    ind = np.where(tmask & zmask)[0]

    ##plot the observations as scattered data over c.grid
    try:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        if obs_rec['is_vector']:
            obs_u = obs_seq['obs'][0,...][ind]
            obs_v = obs_seq['obs'][1,...][ind]
            obs = np.array([obs_u, obs_v])
            c.grid.plot_scatter(ax[0], obs, vmin, vmax, nlevels, x=obs_x[ind], y=obs_y[ind], is_vector=True, units=obs_rec['units'])
            adjust_ax_size(ax[0])

            obs_prior_u = obs_prior_seq[0,...][ind]
            obs_prior_v = obs_prior_seq[1,...][ind]
            obs_prior = np.array([obs_prior_u, obs_prior_v])
            obs_diff = obs - obs_prior
            c.grid.plot_scatter(ax[1], obs_diff, vmin_diff, vmax_diff, nlevels_diff, x=obs_x[ind], y=obs_y[ind], is_vector=True, units=obs_rec['units'])
            adjust_ax_size(ax[1])

        else:
            obs = obs_seq['obs'][ind]
            c.grid.plot_scatter(ax[0], obs, vmin, vmax, nlevels, cmap=cmap, markersize=10, x=obs_x[ind], y=obs_y[ind])
            add_colorbar(fig, ax[0], cmap, vmin, vmax, nlevels, units=obs_rec['units'])

            obs_prior = obs_prior_seq[ind]
            obs_diff = obs - obs_prior
            c.grid.plot_scatter(ax[1], obs_diff, vmin_diff, vmax_diff, nlevels_diff, cmap=cmap_diff, markersize=10, x=obs_x[ind], y=obs_y[ind])
            add_colorbar(fig, ax[1], cmap_diff, vmin_diff, vmax_diff, nlevels_diff, units=obs_rec['units'])

        for i in range(2):
            c.grid.plot_land(ax[i], color=landcolor)
            ax[i].set_xlabel('x (m)', fontsize=14)
            ax[i].set_ylabel('y (m)', fontsize=14)

        ax[0].set_title(f"Observation", fontsize=14)
        ax[1].set_title(f"Diff(Observation - Model member {member+1})", fontsize=14)
        fig.suptitle(f"{obs_rec['dataset_src']}_{obs_rec['name']}, level {k}, {t} ~ {t+dt*dt1h}", fontsize=16)
        plt.savefig(figfile)
        plt.close()

    except Exception as e:
        print(f"ERROR: Failed to plot {obs_rec['name']} at level {k} and time {t} ~ {t+dt*dt1h}")
        raise e

def generate_viewer_html(c, plot_dir, figsize, **kwargs) -> None:
    """Generating a html page to help viewing the ensemble state variables"""
    if c.debug:
        print(f"Generating viewer.html page in {plot_dir}")

    with open(os.path.join(os.path.dirname(__file__), 'viewer.html'), 'rt') as f:
        html_page = f.read()

    variables = ensure_list(kwargs['variables'])
    dataset_src = ensure_list(kwargs['dataset_src'])
    obs_window_min = kwargs.get('obs_window_min', 0)
    obs_window_max = kwargs.get('obs_window_max', 0)
    obs_dt = ensure_list(kwargs['obs_dt'])
    obs_kmin = ensure_list(kwargs['obs_kmin'])
    obs_kmax = ensure_list(kwargs['obs_kmax'])

    levels_by_variable = ""
    times_by_variable = ""
    for i, vname in enumerate(variables):
        name = f"{dataset_src[i]}_{vname}"
        obs_ts = c.time + np.arange(obs_window_min, obs_window_max, obs_dt[i]) * dt1h
        levels = np.arange(obs_kmin[i], obs_kmax[i]+1)

        levels_by_variable += f"'{name}': ["
        for level in levels:
            levels_by_variable += f"{level}, "
        levels_by_variable += "], \n"

        times_by_variable += f"'{name}': ["
        for t in obs_ts:
            times_by_variable += f"'{t:%Y%m%dT%H%M%S}_{t+obs_dt[i]*dt1h:%Y%m%dT%H%M%S}', "
        times_by_variable += "], \n"
    html_page = html_page.replace("LEVELS_BY_VARIABLE", levels_by_variable)
    html_page = html_page.replace("TIMES_BY_VARIABLE", times_by_variable)

    members = "["
    for m in range(c.nens):
        members += f"'{m+1:03}', "
    members += "]"
    html_page = html_page.replace("MEMBERS", members)

    html_page = html_page.replace("TITLE", "Observations")
    html_page = html_page.replace("IMAGE_WIDTH", f"{figsize[0]*60}")
    html_page = html_page.replace("IMAGE_HEIGHT", f"{figsize[1]*60}")

    ##write the html page to file
    with open(os.path.join(plot_dir, 'index.html'), 'w') as f:
        f.write(html_page)
