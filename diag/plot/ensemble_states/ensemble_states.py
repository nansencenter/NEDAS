"""Diagnostic module to plot the ensemble states"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.conversion import ensure_list, t2h, h2t, dt1h
from utils.dir_def import analysis_dir, forecast_dir
from utils.shell_utils import makedir
from assim_tools.state import read_field, read_state_info

def get_task_list(c, **kwargs) -> list:

    variables = ensure_list(kwargs['variables'])
    vmin = ensure_list(kwargs['vmin'])
    vmax = ensure_list(kwargs['vmax'])
    cmap = ensure_list(kwargs['cmap'])
    model_src = kwargs['model_src']

    c.next_time = c.time + c.cycle_period * dt1h

    tasks = []
    for member in range(c.nens):
        for i, vname in enumerate(variables):
            model = c.model_config[model_src]
            levels = model.variables[vname]['levels']
            for k in levels:
                for t in np.arange(t2h(c.time), t2h(c.next_time), model.output_dt):
                    tasks.append({**kwargs, 'time':h2t(t), 'member':member, 'vname':vname, 'k':k, 'vmin':vmin[i], 'vmax':vmax[i], 'cmap':cmap[i]})
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

    figsize = (kwargs.get('fig_size_x', 10), kwargs.get('fig_size_y', 10))
    landcolor = kwargs.get('land_color', None)

    variables = ensure_list(kwargs['variables'])
    vname = kwargs['vname']
    vmin = kwargs['vmin']
    vmax = kwargs['vmax']

    if kwargs['cmap'].split('.')[0] == 'cmocean':
        import cmocean
        cmap = getattr(cmocean.cm, kwargs['cmap'].split('.')[-1])
    else:
        cmap = kwargs['cmap']

    member = kwargs['member']
    k = kwargs['k']
    time = kwargs['time']
    model_src = kwargs['model_src']

    if c.debug:
        print(f"PID {c.pid:4} plotting state variable '{vname:20}' k={k:3} at {time} for member{member+1:03}", flush=True)

    ##if the viewer html file does not exist, generate it
    viewer = os.path.join(plot_dir, 'viewer.html')
    if not os.path.exists(viewer):
        generate_viewer_html(c, plot_dir, model_src, variables)

    ##plot the variables defined in kwargs, save to figfile
    figfile = os.path.join(plot_dir, f"{vname}_k{k}_{time:%Y%m%dT%H%M%S}_mem{member+1:03}.png")

    ##read the field from binary file in analysis directory
    # binfile = os.path.join(analysis_dir(c, c.time), "prior_state.bin")
    # info = read_state_info(binfile)
    # rec_id = None
    # for i, rec in info['fields'].items():
    #     if rec['name'] == vname and rec['k'] == k:
    #         rec_id = i
    #         break
    # if rec_id is None:
    #     raise ValueError("Error: record id not found for variable {vname} in {binfile}")
    # var = read_field(binfile, info, c.mask, member, rec_id)
    # grid = c.grid

    ##read the field from model restart files
    model = c.model_config[model_src]
    if 'forecast_dir' in kwargs:
        fdir = kwargs['forecast_dir'].format(time=c.time)
    else:
        fdir = forecast_dir(c, c.time, model_src)
    var = model.read_var(path=fdir, name=vname, k=k, member=member, time=time)
    grid = model.grid
    
    rec = model.variables[vname]

    ##plot the field
    try:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if rec['is_vector']:
            grid.plot_vectors(ax, var, V=vmax, showref=True, ref_units=rec['units'])
        else:
            im = grid.plot_field(ax, var, vmin=vmin, vmax=vmax, cmap=cmap)
            cbar = plt.colorbar(im, fraction=0.025, pad=0.02)
            cbar.ax.set_title(rec['units'], loc='center', pad=10)
        grid.plot_land(ax, color=landcolor)
        plt.title(f"{vname}, level {k}, {time}, member {member+1}", fontsize=15)
        plt.xlabel('x (m)', fontsize=12)
        plt.ylabel('y (m)', fontsize=12)
        plt.savefig(figfile)
        plt.close()
    except Exception as e:
        print(f"ERROR: Failed to plot {vname} at level {k} and time {time} for member {member+1}")
        raise e

def generate_viewer_html(c, plot_dir, model_src, variables) -> None:
    """Generating a html page to help viewing the ensemble state variables"""
    if c.debug:
        print(f"Generating viewer.html page in {plot_dir}")

    with open(os.path.join(os.path.dirname(__file__), 'viewer.html'), 'rt') as f:
        html_page = f.read()

    ##replace the placeholder with the list of variables,levels,times,members
    levels_by_variable = ""
    for vname in variables:
        levels_by_variable += f"{vname}: ["
        model = c.model_config[model_src]
        for level in model.variables[vname]['levels']:
            levels_by_variable += f"{level}, "
        levels_by_variable += "], \n"
    html_page = html_page.replace("LEVELS_BY_VARIABLE", levels_by_variable)

    times = "["
    for t in np.arange(t2h(c.time_start), t2h(c.time_end), model.output_dt):
        times += f"'{h2t(t):%Y%m%dT%H%M%S}', "
    times += "]"
    html_page = html_page.replace("TIMES", times)

    members = "["
    for m in range(c.nens):
        members += f"'{m+1:03}', "
    members += "]"
    html_page = html_page.replace("MEMBERS", members)

    ##write the html page to file
    with open(os.path.join(plot_dir, 'viewer.html'), 'w') as f:
        f.write(html_page)
