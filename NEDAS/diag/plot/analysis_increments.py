"""Diagnostic module to plot the ensemble states"""

import os
import numpy as np
import matplotlib.pyplot as plt
from NEDAS.utils.conversion import ensure_list, dt1h
from NEDAS.utils.shell_utils import makedir
from NEDAS.utils.graphics import add_colorbar, adjust_ax_size, get_cmap
from NEDAS.assim_tools.state import State

def get_task_list(c, **kwargs) -> list:

    variables = ensure_list(kwargs['variables'])
    vmin_diff = ensure_list(kwargs['vmin_diff'])
    vmax_diff = ensure_list(kwargs['vmax_diff'])
    nlevels_diff = ensure_list(kwargs['nlevels_diff'])
    cmap_diff = ensure_list(kwargs['cmap_diff'])

    state = State(c)

    tasks = []
    for member in range(c.nens):
        for i, vname in enumerate(variables):
            levels = [r['k'] for id,r in state.info['fields'].items() if r['name']==vname]
            assert len(levels)>0, f"cannot find state variable '{vname}'"
            for k in levels:
                for t in c.time + np.array(c.state_time_steps) * dt1h:
                    tasks.append({**kwargs, 'time':t, 'member':member, 'vname':vname, 'k':k, 'vmin_diff':vmin_diff[i], 'vmax_diff':vmax_diff[i], 'nlevels_diff':nlevels_diff[i], 'cmap_diff':cmap_diff[i]})
    return tasks

def run(c, **kwargs) -> None:
    """
    Run diagnostics: plot the ensemble states
    """
    if 'plot_dir' in kwargs:
        plot_dir = kwargs['plot_dir']
    else:
        plot_dir = os.path.join(c.work_dir, 'plots', 'analysis_increments')
    makedir(plot_dir)

    figsize = (kwargs.get('fig_size_x', 9), kwargs.get('fig_size_y', 8))
    landcolor = kwargs.get('land_color', 'gray')

    variables = ensure_list(kwargs['variables'])
    vname = kwargs['vname']
    vmin_diff = kwargs['vmin_diff']
    vmax_diff = kwargs['vmax_diff']
    nlevels_diff = kwargs['nlevels_diff']
    cmap_diff = get_cmap(kwargs['cmap_diff'])

    member = kwargs['member']
    k = kwargs['k']
    time = kwargs['time']

    state = State(c)
    rec_query = [id for id,r in state.info['fields'].items() if r['name']==vname and r['k']==k]
    assert len(rec_query)>0, f"cannot find state variable '{vname}' at k={k}"
    rec_id = rec_query[0]
    rec = state.info['fields'][rec_id]

    if c.debug:
        print(f"PID {c.pid:4} plotting state variable '{vname:20}' k={k:3} at {time} for member{member+1:03}", flush=True)

    ##if the viewer html file does not exist, generate it
    viewer = os.path.join(plot_dir, 'index.html')
    if not os.path.exists(viewer):
        generate_viewer_html(c, plot_dir, variables, figsize)

    ##plot the variables defined in kwargs, save to figfile
    figfile = os.path.join(plot_dir, f"{vname}_k{k}_{time:%Y%m%dT%H%M%S}_mem{member+1:03}.png")

    ##read the field from bin file in analysis dir
    var_prior = state.read_field(state.prior_file, c.grid.mask, member, rec_id)
    var_post = state.read_field(state.post_file, c.grid.mask, member, rec_id)
    incr = var_post - var_prior

    ##plot the field
    try:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if rec['is_vector']:
            c.grid.plot_vectors(ax, incr, V=vmax_diff, showref=True, ref_units=rec['units'])
            adjust_ax_size(ax)

        else:
            c.grid.plot_field(ax, incr, vmin=vmin_diff, vmax=vmax_diff, cmap=cmap_diff)
            add_colorbar(fig, ax, cmap_diff, vmin_diff, vmax_diff, nlevels_diff, units=rec['units'])

        c.grid.plot_land(ax, color=landcolor)
        ax.set_title(f'analysis increment, member {member+1}', fontsize=16)
        ax.set_xlabel('x (m)', fontsize=14)
        ax.set_ylabel('y (m)', fontsize=14)
        fig.suptitle(f"{vname}, level {k:2}, {time}", fontsize=16)
        plt.savefig(figfile)
        plt.close()

    except Exception as e:
        print(f"ERROR: Failed to plot {vname} at level {k} and time {time} for member {member+1}")
        raise e

def generate_viewer_html(c, plot_dir, variables, figsize) -> None:
    """Generating a html page to help viewing the ensemble state variables"""
    if c.debug:
        print(f"Generating viewer.html page in {plot_dir}")

    with open(os.path.join(os.path.dirname(__file__), 'viewer.html'), 'rt') as f:
        html_page = f.read()

    state = State(c)

    ##replace the placeholder with the list of variables,levels,times,members
    levels_by_variable = ""
    times_by_variable = ""
    for vname in variables:
        levels_by_variable += f"'{vname}': ["
        levels = [r['k'] for id,r in state.info['fields'].items() if r['name']==vname]
        levels.sort()
        for level in levels:
            levels_by_variable += f"{level}, "
        levels_by_variable += "], \n"

        times_by_variable += f"'{vname}': ["
        for t in c.time + np.array(c.state_time_steps) * dt1h:
            times_by_variable += f"'{t:%Y%m%dT%H%M%S}', "
        times_by_variable += "], \n"

    html_page = html_page.replace("LEVELS_BY_VARIABLE", levels_by_variable)
    html_page = html_page.replace("TIMES_BY_VARIABLE", times_by_variable)

    members = "["
    for m in range(c.nens):
        members += f"'{m+1:03}', "
    members += "]"
    html_page = html_page.replace("MEMBERS", members)

    html_page = html_page.replace("TITLE", "Analysis Increments in Ensemble States")
    html_page = html_page.replace("IMAGE_WIDTH", f"{figsize[0]*60}")
    html_page = html_page.replace("IMAGE_HEIGHT", f"{figsize[1]*60}")

    ##write the html page to file
    with open(os.path.join(plot_dir, 'index.html'), 'w') as f:
        f.write(html_page)
