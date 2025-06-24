"""Diagnostic module to plot the ensemble states"""

import os
import numpy as np
import matplotlib.pyplot as plt
from NEDAS.utils.conversion import ensure_list, t2h, h2t, dt1h
from NEDAS.utils.shell_utils import makedir
from NEDAS.utils.graphics import add_colorbar, adjust_ax_size, get_cmap

def get_task_list(c, **kwargs) -> list:

    variables = ensure_list(kwargs['variables'])
    model_src = ensure_list(kwargs['model_src'])
    vmin = ensure_list(kwargs['vmin'])
    vmax = ensure_list(kwargs['vmax'])
    nlevels = ensure_list(kwargs['nlevels'])
    cmap = ensure_list(kwargs['cmap'])

    tasks = []
    for member in range(c.nens):
        for i, vname in enumerate(variables):
            model = c.models[model_src[i]]
            levels = model.variables[vname]['levels']
            dt = model.variables[vname]['dt']
            for k in levels:
                for t in np.arange(t2h(c.time), t2h(c.next_time), dt):
                    tasks.append({**kwargs, 'time':h2t(t), 'member':member, 'model_src':model_src[i], 'vname':vname, 'k':k, 'vmin':vmin[i], 'vmax':vmax[i], 'nlevels':nlevels[i], 'cmap':cmap[i]})
    return tasks

def run(c, **kwargs) -> None:
    """
    Run diagnostics: plot the ensemble states
    """
    if 'plot_dir' in kwargs:
        plot_dir = kwargs['plot_dir']
    else:
        plot_dir = os.path.join(c.work_dir, 'plots', 'ensemble_states')
    makedir(plot_dir)

    figsize = (kwargs.get('fig_size_x', 9), kwargs.get('fig_size_y', 8))
    landcolor = kwargs.get('land_color', 'gray')

    variables = ensure_list(kwargs['variables'])
    vname = kwargs['vname']
    vmin = kwargs['vmin']
    vmax = kwargs['vmax']
    nlevels = kwargs['nlevels']
    cmap = get_cmap(kwargs['cmap'])

    member = kwargs['member']
    k = kwargs['k']
    time = kwargs['time']
    model_src = kwargs['model_src']

    if c.debug:
        print(f"PID {c.pid:4} plotting state variable '{vname:20}' k={k:3} at {time} for member{member+1:03}", flush=True)

    ##if the viewer html file does not exist, generate it
    viewer = os.path.join(plot_dir, 'index.html')
    if not os.path.exists(viewer):
        generate_viewer_html(c, plot_dir, model_src, variables, figsize)

    ##plot the variables defined in kwargs, save to figfile
    figfile = os.path.join(plot_dir, f"{vname}_k{k}_{time:%Y%m%dT%H%M%S}_mem{member+1:03}.png")

    ##read the field from model restart files
    model = c.models[model_src]
    if 'forecast_dir' in kwargs:
        fdir = kwargs['forecast_dir'].format(time=c.time)
    else:
        fdir = c.forecast_dir(c.time, model_src)
    model.read_grid(path=fdir, name=vname, k=k, member=member, time=time)
    var = model.read_var(path=fdir, name=vname, k=k, member=member, time=time)
    grid = model.grid

    rec = model.variables[vname]

    ##plot the field
    try:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if rec['is_vector']:
            grid.plot_vectors(ax, var, V=vmax, showref=True, ref_units=rec['units'])
            adjust_ax_size(ax)
        else:
            grid.plot_field(ax, var, vmin=vmin, vmax=vmax, cmap=cmap)
            add_colorbar(fig, ax, cmap, vmin, vmax, nlevels, units=rec['units'])
        grid.plot_land(ax, color=landcolor)
        ax.set_title(f'member {member+1}', fontsize=16)
        ax.set_xlabel('x (m)', fontsize=14)
        ax.set_ylabel('y (m)', fontsize=14)
        fig.suptitle(f"{vname}, level {k:2}, {time}", fontsize=16)
        plt.savefig(figfile)
        plt.close()
    except Exception as e:
        print(f"ERROR: Failed to plot {vname} at level {k} and time {time} for member {member+1}")
        raise e

def generate_viewer_html(c, plot_dir, model_src, variables, figsize) -> None:
    """Generating a html page to help viewing the ensemble state variables"""
    if c.debug:
        print(f"Generating viewer.html page in {plot_dir}")

    with open(os.path.join(os.path.dirname(__file__), 'viewer.html'), 'rt') as f:
        html_page = f.read()

    ##replace the placeholder with the list of variables,levels,times,members
    levels_by_variable = ""
    times_by_variable = ""
    for vname in variables:
        levels_by_variable += f"'{vname}': ["
        model = c.models[model_src]
        for level in model.variables[vname]['levels']:
            levels_by_variable += f"{level}, "
        levels_by_variable += "], \n"
        times_by_variable += f"'{vname}': ["
        for t in np.arange(t2h(c.time_start), t2h(c.time_end), model.variables[vname]['dt']):
            times_by_variable += f"'{h2t(t):%Y%m%dT%H%M%S}', "
        times_by_variable += "], \n"
    html_page = html_page.replace("LEVELS_BY_VARIABLE", levels_by_variable)
    html_page = html_page.replace("TIMES_BY_VARIABLE", times_by_variable)

    members = "["
    for m in range(c.nens):
        members += f"'{m+1:03}', "
    members += "]"
    html_page = html_page.replace("MEMBERS", members)

    html_page = html_page.replace("TITLE", "Ensemble States")
    html_page = html_page.replace("IMAGE_WIDTH", f"{figsize[0]*60}")
    html_page = html_page.replace("IMAGE_HEIGHT", f"{figsize[1]*60}")

    ##write the html page to file
    with open(os.path.join(plot_dir, 'index.html'), 'w') as f:
        f.write(html_page)
