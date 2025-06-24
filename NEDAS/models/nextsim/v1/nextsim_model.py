import os
import glob
import numpy as np
from datetime import datetime, timedelta
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.utils.conversion import t2s, s2t, dt1h, units_convert
from NEDAS.utils.shell_utils import run_command, makedir, run_job
from NEDAS.utils.progress import watch_files
from NEDAS.grid import Grid
from NEDAS.models import Model
from .gmshlib import read_mshfile, proj
from .bin_io import read_data, write_data
from .drift_utils import get_deformation_nodes
from .namelist import namelist

class NextsimModel(Model):
    """
    Class for configuring and running the nextsim v1 model (lagrangian version)
    """
    def __init__(self, config_file=None, parse_args=False, **kwargs):

        super().__init__(config_file, parse_args, **kwargs)

        ##Note: we only work with restart files, normal nextsim binfile have some variables names that
        ##are different from restart files, e.g. Concentration instead of M_conc
        self.native_variables = {
            'seaice_conc': {'name':'M_conc', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':1 },
            'seaice_thick': {'name':'M_thick', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'m' },
            'seaice_velocity': {'name':'M_VT', 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':[0], 'units':'m/s' },
            'seaice_damage': {'name':'M_damage', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':1 },
            'seaice_ridge_ratio': {'name':'M_ridge_ratio', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':1 },
            'seaice_conc_young': {'name':'M_conc_young', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':1 },
            'seaice_thick_young': {'name':'M_h_young', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'m' },
            'seaice_age': {'name':'M_age', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'year' },
            'seaice_conc_myi': {'name':'M_conc_myi', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':1 },
            'seaice_thick_myi': {'name':'M_thick_myi', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':1 },
            'snow_thick': {'name':'M_snow_thick', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'m' },
            'snow_thick_young': {'name':'M_hs_young', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'m' },
            }
        self.diag_variables = {
            'seaice_drift': {'name':'drift', 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':[0], 'units':'km/day' },
            'seaice_deform_div': {'name':'deform_e1', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'1/day' },
            'seaice_deform_shear': {'name':'deform_e2', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'1/day' },
            'seaice_deform_vort': {'name':'deform_e3', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'1/day' },
            }
        self.atmos_forcing_variables = {
            'atmos_surf_velocity': {'name':('x_wind_10m', 'y_wind_10m'), 'is_vector':True, 'dt':self.forcing_dt, 'levels':[0], 'units':'m/s'},
            'atmos_surf_temp': {'name':'air_temperature_2m', 'is_vector':False, 'dt':self.forcing_dt, 'levels':[0], 'units':'K'},
            'atmos_surf_dew_temp': {'name':'dew_point_temperature_2m', 'is_vector':False, 'dt':self.forcing_dt, 'levels':[0], 'units':'K'},
            'atmos_surf_press': {'name':'atm_pressure', 'is_vector':False, 'dt':self.forcing_dt, 'levels':[0], 'units':'Pa'},
            'atmos_precip': {'name':'total_precipitation_rate', 'is_vector':False, 'dt':self.forcing_dt, 'levels':[0], 'units':'kg/m2/s'},
            'atmos_snowfall': {'name':'snowfall_rate', 'is_vector':False, 'dt':self.forcing_dt, 'levels':[0], 'units':'kg/m2/s'},
            'atmos_down_shortwave': {'name':'instantaneous_downwelling_shortwave_radiation', 'is_vector':False, 'dt':self.forcing_dt, 'levels':[0], 'units':'W/m2'},
            'atmos_down_longwave': {'name':'instantaneous_downwelling_longwave_radiation', 'is_vector':False, 'dt':self.forcing_dt, 'levels':[0], 'units':'W/m2'},
            }
        self.variables = {**self.native_variables, **self.diag_variables, **self.atmos_forcing_variables}

        self.grid = None
        self.grid_bank = {}

    def filename(self, **kwargs):
        """
        Get the filename with specified variable name, time, member, etc. 
        """
        kwargs = super().parse_kwargs(**kwargs)
        if kwargs['member'] is not None:
            mstr = '{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        if kwargs['name'] in {**self.native_variables, **self.diag_variables}:
            if kwargs['time'] is not None:
                tstr = kwargs['time'].strftime('%Y%m%dT%H%M%SZ')
                return os.path.join(kwargs['path'], mstr, self.restart_input_path, 'field_'+tstr+'.bin')

            else:
                tstr = '*'
                search = os.path.join(kwargs['path'], mstr, self.restart_input_path, 'field_'+tstr+'.bin')
                flist = glob.glob(search)
                assert len(flist)>0, 'no matching files found: '+search
                return flist[0]   ##return the first matching file

        elif kwargs['name'] in self.atmos_forcing_variables:
            return os.path.join(kwargs['path'], mstr, "data", self.atmos_forcing_path, "generic_ps_atm_"+kwargs['time'].strftime('%Y%m%d')+".nc")

    def read_grid_from_mshfile(self, mshfile):
        """
        Read mshfile and update the self.grid object
        """
        grid_info = read_mshfile(mshfile)
        x, y = grid_info['nodes_x'], grid_info['nodes_y']
        triangles = np.array([np.array(el.node_indices) for el in grid_info['triangles']])
        self.grid = Grid(proj, x, y, regular=False, triangles=triangles)
        self.edges = np.array([np.array(el.node_indices) for el in grid_info['edges']])

    def read_grid(self, **kwargs):
        """
        Update self.grid object based on input kwargs
        """
        if kwargs is None:
            self.read_grid_from_mshfile(os.path.join(self.nextsim_mesh_dir, self.msh_filename))

        kwargs = super().parse_kwargs(**kwargs)
        if kwargs['name'] in {**self.native_variables, **self.diag_variables}:
            if 'meshfile' in kwargs:
                meshfile = kwargs['meshfile']
            else:
                meshfile = self.filename(**kwargs).replace('field', 'mesh')
            ###only need to read the uniq grid once, store known meshfile in memory bank
            if meshfile not in self.grid_bank:
                ##read the grid from mesh file and add to grid_bank
                x = read_data(meshfile, 'Nodes_x')
                y = read_data(meshfile, 'Nodes_y')
                elements = read_data(meshfile, 'Elements')
                n_elements = int(elements.size/3)
                triangles = elements.reshape((n_elements, 3)) - 1
                self.grid_bank[meshfile] = Grid(proj, x, y, regular=False, triangles=triangles)
                self.grid_bank[meshfile].id = read_data(meshfile, 'id')
            self.grid = self.grid_bank[meshfile]

        elif kwargs['name'] in self.atmos_forcing_variables:
            self.grid = Grid.regular_grid(proj, -2.5e6, 2.498e6, -2e6, 2.5e6, 3e3, centered=True)

        self.grid.mask = np.full(self.grid.x.shape, False)  ##no grid points should be masked

    def write_grid(self, **kwargs):
        """
        write updated mesh back to mesh file

        Note: now we assume that number of mesh elements and their indices doesn't change!
        only updating the mesh node position x,y
        """
        kwargs = super().parse_kwargs(**kwargs)
        meshfile = self.filename(**kwargs).replace('field', 'mesh')

        write_data(meshfile, 'Nodes_x', self.grid.x)
        write_data(meshfile, 'Nodes_y', self.grid.y)

        elements = (self.grid.tri.triangles + 1).flatten()
        write_data(meshfile, 'Elements', elements)

    def get_boundary_nodes(self):
        edges = set()
        for triangle in self.grid.tri.triangles:
            for i in range(3):
                edge = tuple(sorted((triangle[i], triangle[(i+1) % 3])))
                if edge in edges:
                    edges.remove(edge)  # If seen twice, it's an internal edge
                else:
                    edges.add(edge)
        return np.unique(np.array(list(edges)))  ##the boundary node indices

    def get_neighbor_nodes(self, nodes):
        neighbors = set()
        for i in nodes:
            r,c = np.where(self.grid.tri.edges == i)
            for n in self.grid.tri.edges[r, c-1]:
                neighbors.add(n)
        return neighbors

    def taper_boundary(self, fld, depth=5):
        fld_taper = fld.copy()
        used_nodes = []
        nodes = self.get_boundary_nodes()
        fld_taper[nodes] = 0
        used_nodes.extend(nodes)
        for n in range(1, depth):
            nodes = self.get_neighbor_nodes(nodes)
            nodes.difference_update(used_nodes)
            nodes = list(set(nodes))
            fld_taper[nodes] *= n / depth
            used_nodes.extend(nodes)
        return fld_taper

    def displace(self, u, v, **kwargs):
        """
        Nextsim has a Lagrangian mesh, so it's possible to displace the mesh coordinates directly
        Inputs:
        - u, v: displacement vectors defined on self.grid.x,y
        """
        ##read grid, refresh self.grid.x, y
        self.read_grid(**kwargs)

        ##make sure boundary is not moving
        u = self.taper_boundary(u)
        v = self.taper_boundary(v)

        ##apply the displacement vectors -u, -v
        self.grid.x += u
        self.grid.y += v

        ##write the updated mesh node xy to the restart file
        self.write_grid(**kwargs)

    def read_mask(self, **kwargs):
        pass

    def prepare_mask(self, dst_grid):
        self.grid.set_destination_grid(dst_grid)
        tmp = self.grid.convert(np.ones(self.grid.x.shape))
        return np.isnan(tmp)

    def read_var(self, **kwargs):
        """read variable from a model restart file"""
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        if name in self.native_variables:
            var = read_data(fname, rec['name'])
            ##nextsim restart file concatenates u,v component, so reshape if is_vector
            if rec['is_vector']:
                var = var.reshape((2, -1))

        elif name in self.diag_variables:
            var = self.get_diag_var(**kwargs)

        elif name in self.atmos_forcing_variables:
            time = kwargs['time']
            nt_in_file = int(np.round(time.hour / rec['dt']))
            if rec['is_vector']:
                u = nc_read_var(fname, rec['name'][0])[nt_in_file, ...]
                v = nc_read_var(fname, rec['name'][1])[nt_in_file, ...]
                var = np.array([u, v])
            else:
                var = nc_read_var(fname, rec['name'])[nt_in_file, ...]

        ##convert units if native unit is not the same as required by kwargs
        var = units_convert(rec['units'], kwargs['units'], var)
        return var

    def write_var(self, var, **kwargs):
        """write variable back to a model restart file"""
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        ##convert units back if necessary
        var = units_convert(kwargs['units'], rec['units'], var)

        if name in self.native_variables:
            ##nextsim restart file concatenates u,v component, so flatten if is_vector
            if rec['is_vector']:
                var = var.flatten()
            ##check if original var is on mesh nodes or elements
            # var_orig = read_data(fname, rec['name']).flatten()
            # if var_orig.size != var.size:
            #     ##the grid.convert interpolate to nodes by default, if size mismatch, this means
            #     ##we need element values, take the average of the node values here
            #     var = np.nanmean(var[grid.tri.triangles], axis=1)

            ##output the var to restart file
            write_data(fname, rec['name'], var)

        elif name in self.atmos_forcing_variables:
            time = kwargs['time']
            nt_in_file = int(np.round(time.hour / rec['dt']))
            ny, nx = var.shape[-2:]
            if rec['is_vector']:
                for i in range(2):
                    data_attr={'standard_name':rec['name'][i], 'units':rec['units'], 'grid_mapping':'projection_stereo'}
                    nc_write_var(fname, {'time':None, 'y':ny, 'x':nx}, rec['name'][i], var[i, ...], recno={'time':nt_in_file}, attr=data_attr, comm=kwargs['comm'])
            else:
                data_attr={'standard_name':rec['name'], 'units':rec['units'], 'grid_mapping':'projection_stereo'}
                nc_write_var(fname, {'time':None, 'y':ny, 'x':nx}, rec['name'], var, recno={'time':nt_in_file}, attr=data_attr, comm=kwargs['comm'])

    def z_coords(self, **kwargs):
        ##for nextsim, just discard inputs and simply return zero as z_coords
        return np.zeros(self.grid.x.shape)

    def get_seaice_drift(self, **kwargs):
        dt1day = timedelta(days=1)
        t2 = kwargs['time']
        t1 = kwargs['time'] - dt1day*3  ##TODO: make the duration configurable
        dt = (t2 - t1) / dt1day

        if kwargs['member'] is not None:
            mstr = '{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        meshfile = os.path.join(kwargs['path'], '..', '..', t2s(t1), 'nextsim.v1', mstr, 'restart', f"mesh_{t1:%Y%m%dT%H%M%SZ}.bin")
        self.read_grid(meshfile=meshfile, **kwargs)
        grid1 = self.grid_bank[meshfile]
        meshfile = os.path.join(kwargs['path'], '..', '..', t2s(t2-dt1day), 'nextsim.v1', mstr, 'restart', f"mesh_{t2:%Y%m%dT%H%M%SZ}.bin")
        self.read_grid(meshfile=meshfile, **kwargs)
        grid2 = self.grid_bank[meshfile]
        ids_cmn_12, ids1i, ids2i = np.intersect1d(grid1.id, grid2.id, return_indices=True)
        x1n = grid1.x[ids1i] / 1000
        y1n = grid1.y[ids1i] / 1000
        x2n = grid2.x[ids2i] / 1000
        y2n = grid2.y[ids2i] / 1000
        u = (x2n - x1n) / dt
        v = (y2n - y1n) / dt
        self.grid = Grid(grid1.proj, grid2.x[ids2i], grid2.y[ids2i], regular=False)
        return x2n, y2n, u, v

    def get_diag_var(self, **kwargs):
        name = kwargs['name']
        if name == 'seaice_drift':
            _, _, u, v = self.get_seaice_drift(**kwargs)
            return np.array([u, v])

        if 'seaice_deform' in name:
            x, y, u, v = self.get_seaice_drift(**kwargs)
            e1, e2, e3, _, _, _ = get_deformation_nodes(x, y, u, v)
            if name == 'seaice_deform_div':
                return e1
            if name == 'seaice_deform_shear':
                return e2
            if name == 'seaice_deform_vort':
                return e3

        raise NotImplementedError('cannot get diagnostic variable {name}')

    def read_param(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        return getattr(self, kwargs['name'])

    def write_param(self, param, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        setattr(self, kwargs['name'], param)

    def preprocess(self, task_id=0, **kwargs):
        """Preprocess the dir, collect input files for model run"""
        ##put sequence of operation here to generate the initial condition files for nextsim
        kwargs = super().parse_kwargs(**kwargs)
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        if kwargs['member'] is not None:
            mstr = '{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        run_dir = os.path.join(kwargs['path'], mstr)
        makedir(run_dir)

        ##prepare restart files
        restart_file = self.filename(**{**kwargs, 'path':kwargs['restart_dir']})
        shell_cmd = f"cd {run_dir}; "
        shell_cmd += f"mkdir -p {self.restart_input_path}; cd {self.restart_input_path}; "
        field_bin = restart_file
        field_dat = field_bin.replace('.bin', '.dat')
        mesh_bin = restart_file.replace('field', 'mesh')
        mesh_dat = mesh_bin.replace('.bin', '.dat')
        for file in [field_bin, field_dat, mesh_bin, mesh_dat]:
            shell_cmd += f"cp -fL {file} .; "
        run_command(shell_cmd)

        ##prepare other input data (bathymetry, forcing, etc.) for the model run
        shell_cmd = f"cd {run_dir}; "
        shell_cmd += f"rm -rf data; mkdir -p data; cd data; "
        ##bathymetry data
        shell_cmd += f"ln -fs {os.path.join(self.nextsim_data_dir, 'BATHYMETRY', '*')} .; "
        ##ocean forcing (for now it is just linked over)
        shell_cmd += f"ln -fs {os.path.join(self.nextsim_data_dir, self.ocean_forcing_path)} .; "
        ##atmos forcing (make a copy, later they will be perturbed)
        shell_cmd += f"mkdir -p {self.atmos_forcing_path}; cd {self.atmos_forcing_path}; "
        t = time
        while t <= next_time:
            shell_cmd += f"cp -fL {os.path.join(self.nextsim_data_dir, self.atmos_forcing_path, 'generic_ps_atm_'+t.strftime('%Y%m%d')+'.nc')} .; "
            t += 24 * dt1h  ##forcing files are stored daily
        run_command(shell_cmd)

    def postprocess(self, task_id=0, **kwargs):
        ##place holder for now
        ##for any post processing needed after assimilation, to fix any model state that is not consistent
        kwargs = super().parse_kwargs(**kwargs)
        time = kwargs['time']
        if kwargs['member'] is not None:
            mstr = '{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        run_dir = os.path.join(kwargs['path'], mstr)
        restart_file = self.filename(**kwargs)

        ##read seaice conc and thick, check value, fix values out of normal range, then write back to file
        sic = self.read_var(**{**kwargs, 'name':'seaice_conc', 'units':1})
        sit = self.read_var(**{**kwargs, 'name':'seaice_thick', 'units':'m'})
        damage = self.read_var(**{**kwargs, 'name':'seaice_damage', 'units':1})
        rr = self.read_var(**{**kwargs, 'name':'seaice_ridge_ratio', 'units':1})

        sic = np.maximum(np.minimum(sic, 0.9999), 0.0)
        sit = np.maximum(sit, 0.0)
        damage = np.maximum(np.minimum(damage, 0.9999), 0.0)
        rr = np.maximum(np.minimum(rr, 0.9999), 0.0)

        self.write_var(sic, **{**kwargs, 'name':'seaice_conc', 'units':1})
        self.write_var(sit, **{**kwargs, 'name':'seaice_thick', 'units':'m'})
        self.write_var(damage, **{**kwargs, 'name':'seaice_damage', 'units':1})
        self.write_var(rr, **{**kwargs, 'name':'seaice_ridge_ratio', 'units':1})

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h
        input_file = self.filename(**kwargs)
        output_file = self.filename(**{**kwargs, 'time':next_time})

        if kwargs['member'] is not None:
            mstr = '{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        run_dir = os.path.join(kwargs['path'], mstr)
        makedir(run_dir)

        ##check input files
        field_bin = input_file
        field_dat = field_bin.replace('.bin', '.dat')
        mesh_bin = input_file.replace('field', 'mesh')
        mesh_dat = mesh_bin.replace('.bin', '.dat')
        for file in [field_bin, field_dat, mesh_bin, mesh_dat]:
            if not os.path.exists(file):
                raise RuntimeError("input file is missing: "+file)

        ##build command to run the model
        model_exe = os.path.join(self.nextsim_dir, 'model', 'bin', 'nextsim.exec')
        log_file = os.path.join(run_dir, 'run.log')
        run_command("touch "+log_file)

        shell_cmd = f". {self.model_env}; "
        shell_cmd += f"cd {run_dir}; "
        shell_cmd += f"export NEXTSIM_DATA_DIR={os.path.join(run_dir,'data')}; "
        shell_cmd += f"JOB_EXECUTE {model_exe} --config-files=config/nextsim.cfg >& run.log"

        ##give it several tries, each time decreasing time step
        for dt_ratio in [1, 0.5]:
            ##check output, if success skip further tries
            with open(log_file, 'rt') as f:
                if 'Simulation done' in f.read():
                    break

            self.timestep *= dt_ratio

            ##this creates nextsim.cfg.in in run_dir/config
            ##somehow the new version nextsim doesnt like nextsim.cfg to appear in run_dir
            config_dir = os.path.join(run_dir, 'config')
            makedir(config_dir)
            namelist(self, time, forecast_period, config_dir)

       ##run the model and wait for results
            run_job(shell_cmd, job_name='nextsim.run', run_dir=run_dir,
                    nproc=self.nproc_per_run, offset=task_id*self.nproc_per_run,
                    walltime=self.walltime, **kwargs)

        ##checkout output files
        watch_files([output_file])

