"""Diagnostic module to plot the ensemble states"""

import os
import numpy as np
import matplotlib.pyplot as plt
from NEDAS.grid.grid_2d_base import Grid2DBase
from NEDAS.utils.conversion import ensure_list, dt1h
from NEDAS.utils.graphics import add_colorbar, adjust_ax_size, get_cmap
from NEDAS.core.state import State
from NEDAS.core.context import Context

def get_task_list(c: Context, **kwargs) -> list:

    variables = ensure_list(kwargs['variables'])
    vmin_diff = ensure_list(kwargs['vmin_diff'])
    vmax_diff = ensure_list(kwargs['vmax_diff'])
    nlevels_diff = ensure_list(kwargs['nlevels_diff'])
    cmap_diff = ensure_list(kwargs['cmap_diff'])

    # single State construction — local variable, does not mutate c
    state = State(c)

    # pre-compute per-variable metadata once
    levels_by_var = {}
    times_by_var = {}
    for vname in variables:
        levels_by_var[vname] = sorted({r.k for _, r in state.info.fields.items() if r.name == vname})
        assert len(levels_by_var[vname]) > 0, f"cannot find state variable '{vname}'"
        times_by_var[vname] = list(c.time + np.array(c.config.state_time_steps) * dt1h)

    tasks = []
    for member in range(c.nens):
        for i, vname in enumerate(variables):
            for k in levels_by_var[vname]:
                rec_ids = [id for id, r in state.info.fields.items() if r.name == vname and r.k == k]
                rec_id = rec_ids[0]
                rec = state.info.fields[rec_id].asdict()
                for t in times_by_var[vname]:
                    tasks.append({**kwargs,
                                  'time': t, 'member': member, 'vname': vname, 'k': k,
                                  'rec_id': rec_id, 'rec': rec,
                                  'vmin_diff': vmin_diff[i], 'vmax_diff': vmax_diff[i],
                                  'nlevels_diff': nlevels_diff[i], 'cmap_diff': cmap_diff[i]})

    # generate the viewer HTML once here, before parallel task dispatch
    if 'plot_dir' in kwargs:
        plot_dir = kwargs['plot_dir']
    else:
        plot_dir = os.path.join(c.config.work_dir, 'plots', 'analysis_increments')
    figsize = (kwargs.get('fig_size_x', 9), kwargs.get('fig_size_y', 8))
    generate_viewer_html(c, plot_dir, variables, levels_by_var, times_by_var, figsize)

    return tasks

def run(c: Context, **kwargs) -> None:
    """
    Run diagnostics: plot the ensemble states
    """
    if 'plot_dir' in kwargs:
        plot_dir = kwargs['plot_dir']
    else:
        plot_dir = os.path.join(c.config.work_dir, 'plots', 'analysis_increments')
    c.fs.make_dir(plot_dir)

    figsize = (kwargs.get('fig_size_x', 9), kwargs.get('fig_size_y', 8))
    landcolor = kwargs.get('land_color', 'gray')

    vname = kwargs['vname']
    vmin_diff = kwargs['vmin_diff']
    vmax_diff = kwargs['vmax_diff']
    nlevels_diff = kwargs['nlevels_diff']
    cmap_diff = get_cmap(kwargs['cmap_diff'])

    member = kwargs['member']
    k = kwargs['k']
    time = kwargs['time']
    rec_id = kwargs['rec_id']
    rec = kwargs['rec']

    c.debug_message = f"plotting state variable '{vname:20}' k={k:3} at {time} for member{member+1:03}"

    figfile = os.path.join(plot_dir, f"{vname}_k{k}_{time:%Y%m%dT%H%M%S}_mem{member+1:03}.png")

    # read the field from bin file in analysis dir
    var_prior = c.io.read_field(c, 'prior', rec_id, member)
    var_post = c.io.read_field(c, 'post', rec_id, member)
    incr = var_post - var_prior

    # plot the field
    assert isinstance(c.grid, Grid2DBase), f"{c.grid} is not a 2D Grid"
    try:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if rec['is_vector']:
            c.grid.plot_vectors(ax, incr, V=vmax_diff, showref=True, ref_units=rec['units'])
            adjust_ax_size(ax)
        else:
            c.grid.plot_field(ax, incr, vmin=vmin_diff, vmax=vmax_diff, cmap=cmap_diff)  # type: ignore
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

def generate_viewer_html(c, plot_dir, variables, levels_by_var, times_by_var, figsize) -> None:
    """Generate a static HTML viewer page for browsing the plotted increments."""
    c.debug_message = f"Generating viewer.html page in {plot_dir}"

    with open(os.path.join(os.path.dirname(__file__), 'viewer.html'), 'rt') as f:
        html_page = f.read()

    levels_str = ""
    times_str = ""
    for vname in variables:
        levels_str += f"'{vname}': ["
        for level in levels_by_var[vname]:
            levels_str += f"{level}, "
        levels_str += "], \n"

        times_str += f"'{vname}': ["
        for t in times_by_var[vname]:
            times_str += f"'{t:%Y%m%dT%H%M%S}', "
        times_str += "], \n"

    html_page = html_page.replace("LEVELS_BY_VARIABLE", levels_str)
    html_page = html_page.replace("TIMES_BY_VARIABLE", times_str)

    members = "[" + "".join(f"'{m+1:03}', " for m in range(c.nens)) + "]"
    html_page = html_page.replace("MEMBERS", members)
    html_page = html_page.replace("TITLE", "Analysis Increments in Ensemble States")
    html_page = html_page.replace("IMAGE_WIDTH", f"{figsize[0]*60}")
    html_page = html_page.replace("IMAGE_HEIGHT", f"{figsize[1]*60}")

    c.fs.make_dir(plot_dir)
    with open(os.path.join(plot_dir, 'index.html'), 'w') as f:
        f.write(html_page)
