import os
from functools import lru_cache
from datetime import datetime, timedelta
import numpy as np
from utils.conversion import units_convert, t2s, dt1h
from utils.netcdf_lib import nc_read_var, nc_write_var
from utils.shell_utils import run_command, run_job, makedir
from utils.progress import watch_log, find_keyword_in_file, watch_files
from .namelist import namelist
from .postproc import adjust_dp, stmt_fns_sigma, stmt_fns_kappaf
from .cice_utils import thickness_upper_limit
from ..time_format import dayfor
from ..abfile import ABFileRestart, ABFileArchv, ABFileForcing
from ..model_grid import get_topaz_grid, get_depth, get_mean_ssh, stagger, destagger
from ...model_config import ModelConfig

class Topaz5Model(ModelConfig):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        levels = np.arange(self.kdm) + 1  ##ocean levels, from top to bottom, k=1..kdm
        level_sfc = np.array([0])    ##some variables are only defined on surface level k=0
        level_ncat = np.arange(5)   ##some ice variables have 5 categories, treating them as levels also indexed by k

        self.restart_variables = {
            'ocean_velocity':    {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':levels, 'units':'m/s'},
            'ocean_layer_thick': {'name':'dp', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':levels, 'units':'Pa'},
            'ocean_temp':        {'name':'temp', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':levels, 'units':'C'},
            'ocean_saln':        {'name':'saln', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':levels, 'units':'psu'},
            'ocean_b_velocity':  {'name':('ubavg', 'vbavg'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m/s'},
            'ocean_b_press':     {'name':'pbavg', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'Pa'},
            'ocean_mixl_depth':  {'name':'dpmixl', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'Pa'},
            'ocean_bot_press':   {'name':'pbot', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'Pa'},
            'ocean_bot_dense':   {'name':'thkk', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'?'},
            'ocean_bot_montg_pot': {'name':'psikk', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'?'},
            }

        self.archive_variables = {
            'ocean_velocity_daily': {'name':('u-vel.', 'v-vel.'), 'dtype':'float', 'is_vector':True, 'dt':self.output_dt, 'levels':levels, 'units':'m/s'},
            'ocean_layer_thick_daily': {'name':'thknss', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':levels, 'units':'Pa'},
            'ocean_temp_daily': {'name':'temp', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':levels, 'units':'C'},
            'ocean_saln_daily': {'name':'saln', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':levels, 'units':'psu'},
            'ocean_mixl_depth_daily': {'name':'mix_dpth', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'Pa'},
            'ocean_dense_daily': {'name':'dense', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'?'},
            'ocean_surf_height_daily': {'name':'srfhgt', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'m'},
            }

        self.iced_variables = {
            'seaice_velocity': {'name':('uvel', 'vvel'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m/s'},
            'seaice_conc_ncat':   {'name':'aicen', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_ncat, 'units':'%'},
            'seaice_volume_ncat':  {'name':'vicen', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_ncat, 'units':'m'},
            'snow_volume_ncat':  {'name':'vsnon', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_ncat, 'units':'m'},
            }

        self.iceh_variables = {
            'seaice_velocity_daily': {'name':('uvel_d', 'vvel_d'), 'dtype':'float', 'is_vector':True, 'dt':self.output_dt, 'levels':level_sfc, 'units':'m/s'},
            'seaice_conc_daily': {'name':'aice_d', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'%'},
            'seaice_thick_daily': {'name':'hi_d', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'m'},
            'seaice_surf_temp_daily': {'name':'Tsfc_d', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'C'},
            'seaice_saln_daily': {'name':'sice_d', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'ppt'},
            'snow_thick_daily': {'name':'hs_d', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'m'},
            }

        self.atmos_forcing_variables = {
            'atmos_surf_velocity': {'name':('wndewd', 'wndnwd'), 'dtype':'float', 'is_vector':True, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'m/s'},
            'atmos_surf_temp':     {'name':'airtmp', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'C'},
            'atmos_surf_dewpoint': {'name':'dewpt', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'K'},
            'atmos_surf_press':    {'name':'mslprs', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'Pa'},
            'atmos_precip':        {'name':'precip', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'m/s'},
            'atmos_down_longwave': {'name':'radflx', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'W/m2'},
            'atmos_down_shortwave': {'name':'shwflx', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'W/m2'},
            'atmos_vapor_mix_ratio': {'name':'vapmix', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'kg/kg'},
            }
        self.force_synoptic_names = [name for r in self.atmos_forcing_variables.values() for name in (r['name'] if isinstance(r['name'], tuple) else [r['name']])]

        self.diag_variables = {
            'ocean_surf_height': {'name':'ssh', 'operator':self.get_ocean_surf_height, 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m'},
            'ocean_surf_height_anomaly': {'name':'sla', 'operator':self.get_ocean_surf_height_anomaly, 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m'},
            'ocean_surf_temp': {'name':'sst', 'operator':self.get_ocean_surf_temp, 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'K'},
            'seaice_conc': {'name':'sic', 'operator':self.get_seaice_conc, 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'%'},
            'seaice_thick': {'name':'sit', 'operator':self.get_seaice_thick, 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m'},
            'snow_thick': {'name':'snwt', 'operator':self.get_snow_thick, 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m'},
            }
 
        self.variables = {**self.restart_variables,
                          **self.iced_variables,
                          **self.iceh_variables,
                          **self.atmos_forcing_variables,
                          **self.diag_variables,
                          **self.archive_variables}
               
        ##model grid
        grid_info_file = os.path.join(self.basedir, 'topo', 'grid.info')
        self.grid = get_topaz_grid(grid_info_file)

        self.depthfile = os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}.a')
        self.depth, self.mask = get_depth(self.depthfile, self.grid)

        self.meanssh_file = os.path.join(self.basedir, 'topo', 'meanssh.uf')
        if os.path.exists(self.meanssh_file):
            self.meanssh = get_mean_ssh(self.meanssh_file, self.grid)
        else:
            self.meanssh = None

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        ##filename for each model component
        if kwargs['name'] in self.restart_variables:
            tstr = kwargs['time'].strftime('%Y_%j_%H_0000')
            file = 'restart.'+tstr+mstr+'.a'
            #return os.path.join(kwargs['path'], 'restart.'+tstr+mstr+'.a')

        elif kwargs['name'] in self.iced_variables:
            tstr = kwargs['time'].strftime('%Y-%m-%d-00000')
            file = 'iced.'+tstr+mstr+'.nc'

        elif kwargs['name'] in self.iceh_variables:
            tstr = kwargs['time'].strftime('%Y-%m-%d')
            file = os.path.join(mstr[1:], 'SCRATCH', 'cice', 'iceh.'+tstr+'.nc')

        elif kwargs['name'] in self.atmos_forcing_variables:
            file = os.path.join(mstr[1:], 'SCRATCH', 'forcing')
        
        elif kwargs['name'] in self.archive_variables:
            tstr = kwargs['time'].strftime('%Y_%j_12')  ##archive variables are daily means defined on 12z
            file = os.path.join(mstr[1:], 'SCRATCH', 'archm.'+tstr+'.a')
        
        else:
            raise ValueError(f"filename: ERROR: unknown variable name '{kwargs['name']}'")

        path = kwargs['path']
        fname = os.path.join(path, file)
        if os.path.exists(fname):
            return fname
        
        ##if no corresponding files found under the given path
        ##try to find two layers above for the other cycle time
        dirs = path.split(os.sep)
        if dirs[-1] == 'topaz.v5' and len(dirs[-2])==12:
            root = os.sep.join(dirs[:-2])
            for tdir in os.listdir(root):
                fname = os.path.join(root, tdir, 'topaz.v5', file)
                if os.path.exists(fname):
                    return fname
        raise FileNotFoundError(f"filename: ERROR: could not find {file} in {path} or its parent directories")

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        ##get the variable from restart files
        if name in self.restart_variables:
            f = ABFileRestart(fname, 'r', idm=self.grid.nx, jdm=self.grid.ny)
            if rec['is_vector']:
                var1 = f.read_field(rec['name'][0], level=kwargs['k'], tlevel=1, mask=None)
                var2 = f.read_field(rec['name'][1], level=kwargs['k'], tlevel=1, mask=None)
                var = np.array([var1, var2])
            else:
                var = f.read_field(rec['name'], level=kwargs['k'], tlevel=1, mask=None)
            f.close()

        elif name in self.iced_variables:
            if rec['is_vector']:
                if name[-5:] == '_ncat':  ##ncat variable
                    var1 = nc_read_var(fname, rec['name'][0])[kwargs['k'],...]
                    var2 = nc_read_var(fname, rec['name'][1])[kwargs['k'],...]
                else:
                    var1 = nc_read_var(fname, rec['name'][0])
                    var2 = nc_read_var(fname, rec['name'][1])
                var = np.array([var1, var2])
            else:
                if name[-5:] == '_ncat':  ##ncat variable
                    var = nc_read_var(fname, rec['name'])[kwargs['k'],...]
                else:
                    var = nc_read_var(fname, rec['name'])

        elif name in self.iceh_variables:
            if rec['is_vector']:
                var1 = nc_read_var(fname, rec['name'][0])[0, ...]
                var2 = nc_read_var(fname, rec['name'][1])[0, ...]
                var = np.array([var1, var2])
            else:
                var = nc_read_var(fname, rec['name'])[0, ...]

        elif name in self.atmos_forcing_variables:
            dtime = (kwargs['time'] - datetime(1900,12,31)) / (24*dt1h)
            if rec['is_vector']:
                f1 = ABFileForcing(fname+'.'+rec['name'][0], 'r')
                var1 = f1.read_field(rec['name'][0], dtime)
                f2 = ABFileForcing(fname+'.'+rec['name'][1], 'r')
                var2 = f2.read_field(rec['name'][1], dtime)
                var = np.array([var1, var2])
                f1.close()
                f2.close()
            else:
                f = ABFileForcing(fname+'.'+rec['name'], 'r')
                var = f.read_field(rec['name'], dtime)
                f.close()

        elif name in self.diag_variables:
            var = rec['operator'](**kwargs)

        elif name in self.archive_variables:
            f = ABFileArchv(fname, 'r', mask=True)
            if rec['is_vector']:
                var1 = f.read_field(rec['name'][0], level=kwargs['k'])
                var2 = f.read_field(rec['name'][1], level=kwargs['k'])
                var = np.array([var1, var2])
            else:
                var = f.read_field(rec['name'], level=kwargs['k'])
            f.close()

        ##convert units if necessary
        var = units_convert(kwargs['units'], rec['units'], var)
        return var

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        ##convert back to old units
        var = units_convert(rec['units'], kwargs['units'], var)

        if name in self.restart_variables:
            ##open the restart file for over-writing
            ##the 'r+' mode and a new overwrite_field method were added in the ABFileRestart in .abfile
            f = ABFileRestart(fname, 'r+', idm=self.grid.nx, jdm=self.grid.ny, mask=True)
            if rec['is_vector']:
                for i in range(2):
                    f.overwrite_field(var[i,...], None, rec['name'][i], level=kwargs['k'], tlevel=1)
            else:
                f.overwrite_field(var, None, rec['name'], level=kwargs['k'], tlevel=1)
            f.close()

        elif name in self.iced_variables:
            if rec['is_vector']:
                for i in range(2):
                    if rec['name'][i][-5:] == '_ncat':  ##ncat variable
                        dims = {'ncat':None, 'nj':self.grid.ny, 'ni':self.grid.nx}
                        recno = {'ncat':kwargs['k']}
                    else:
                        dims = {'nj':self.grid.ny, 'ni':self.grid.nx}
                        recno = None
                    nc_write_var(fname, dims, rec['name'][i], var[i,...], recno=recno, comm=kwargs['comm'])
            else:
                if rec['name'][-5:] == '_ncat':  ##ncat variable
                    dims = {'ncat':None, 'nj':self.grid.ny, 'ni':self.grid.nx}
                    recno = {'ncat':kwargs['k']}
                else:
                    dims = {'nj':self.grid.ny, 'ni':self.grid.nx}
                    recno = None
                nc_write_var(fname, dims, rec['name'], var, recno=recno, comm=kwargs['comm'])

        elif name in self.atmos_forcing_variables:
            dtime = (kwargs['time'] - datetime(1900,12,31)) / timedelta(days=1)
            if rec['is_vector']:
                for i in range(2):
                    f = ABFileForcing(fname+'.'+rec['name'][i], 'r+')
                    f.overwrite_field(var[i,...], None, rec['name'][i], dtime)
                    f.close()
            else:
                f = ABFileForcing(fname+'.'+rec['name'], 'r+')
                f.overwrite_field(var, None, rec['name'], dtime)
                f.close()

    @lru_cache(maxsize=3)
    def z_coords(self, **kwargs):
        """
        Calculate vertical coordinates given the 3D model state
        Return:
        - z: np.array
        The corresponding z field
        """
        ##some defaults if not set in kwargs
        if 'k' not in kwargs:
            kwargs['k'] = 0

        z = np.zeros(self.grid.x.shape)

        if kwargs['k'] == 0:
            ##if level index is 0, this is the surface, so just return zeros
            return z

        else:
            ##get layer thickness and convert to units
            rec = kwargs.copy()
            rec['name'] = 'ocean_layer_thick'
            rec['units'] = self.variables['ocean_layer_thick']['units'] ##should be Pa
            if self.z_units == 'm':
                dz = - self.read_var(**rec) / self.ONEM ##in meters, negative relative to surface
            elif self.z_units == 'Pa':
                dz = self.read_var(**rec)
            else:
                raise ValueError('do not know how to calculate z_coords for z_units = '+self.z_units)

            ##use recursive func, get previous layer z and add dz
            kwargs['k'] -= 1
            z_prev = self.z_coords(**kwargs)
            return z_prev + dz

    def get_ocean_surf_height(self, **kwargs):
        """Get ocean surface height from restart variables
        Adapted from p_ssh_from_state.F90 in ReanalysisTP5/SSHFromState_HYCOMICE"""
        tbaric = (self.kapref == self.thflag)

        restart_file = self.filename(**{**kwargs, 'name':'ocean_bot_montg_pot'})
        f = ABFileRestart(restart_file, 'r+', idm=self.grid.nx, jdm=self.grid.ny, mask=True)
        psikk = f.read_field('psikk', level=0, tlevel=1)
        thkk = f.read_field('thkk', level=0, tlevel=1)
        pbavg = f.read_field('pbavg', level=0, tlevel=1)

        ind = (self.depth < -0.1) & ~(np.isnan(self.depth))  ##valid points to calculate ssh on

        levels = list(self.variables['ocean_layer_thick']['levels'])
        idm, jdm, kdm = self.grid.nx, self.grid.ny, len(levels)
        pres = np.zeros((kdm+1, jdm, idm))     # cumulative pressure
        thstar = np.zeros((kdm, jdm, idm))
        montg = np.zeros((jdm, idm))
        oneta = np.zeros((jdm, idm))
        ssh = np.zeros((jdm, idm))

        for k in range(kdm):
            saln = f.read_field('saln', level=levels[k], tlevel=1)
            temp = f.read_field('temp', level=levels[k], tlevel=1)
            dp = f.read_field('dp', level=levels[k], tlevel=1)
            # use upper interface pressure in converting sigma to sigma-star
            # this is to avoid density variations in layers intersected by bottom
            th = stmt_fns_sigma(self.thflag, temp, saln)
            kapf = stmt_fns_kappaf(self.thflag, temp, saln, pres[k, ...], self.thref)
            if tbaric:
                thstar[k, ...][ind] = th[ind] + kapf[ind]
            else:
                thstar[k, ...][ind] = th[ind]
            pres[k+1, ...][ind] = pres[k, ...][ind] + dp[ind]
        oneta[ind] = 1. + pbavg[ind] / pres[-1, ...][ind]
        # m_prime in lowest layer
        montg[ind] = psikk[ind] + (pres[-1, ...][ind] * (thkk[ind] + self.thbase - thstar[-1, ...][ind]) - pbavg[ind] * (thstar[-1, ...][ind])) * self.thref**2
        # m_prime in remaining layers
        for k in range(kdm-2, -1, -1):
            montg[ind] += pres[k+1, ...][ind] * oneta[ind] * (thstar[k+1, ...][ind] - thstar[k, ...][ind]) * self.thref**2
        ssh[ind] = (montg[ind] / self.thref + pbavg[ind]) / self.ONEM
        ssh[~ind] = np.nan
        return ssh

    def get_ocean_surf_height_anomaly(self, **kwargs):
        assert self.meanssh is not None, f"SLA: cannot find meanssh file {self.meanssh_file}"
        ssh = self.get_ocean_surf_height(**kwargs)
        sla = ssh - self.meanssh
        return sla
    
    def get_ocean_surf_temp(self, **kwargs):
        #just return first level ocean_temp
        return self.read_var(**{**kwargs, 'name':'ocean_temp', 'k':1})

    @lru_cache(maxsize=3)
    def get_seaice_conc(self, **kwargs):
        """
        Get total seaice concentration from multicategory ice concentration (aicen)
        adapted from ReanalysisTP5/SSHFromState_HYCOMICE/mod_read_icednc by J. Xie
        """
        seaice_conc = np.zeros(self.grid.x.shape)
        rec = kwargs.copy()
        rec['name'] = 'seaice_conc_ncat'
        rec['units'] = self.variables['seaice_conc_ncat']['units']
        for k in range(len(self.variables['seaice_conc_ncat']['levels'])):
            seaice_conc += self.read_var(**{**rec, 'k':k})
        
        seaice_conc[np.where(seaice_conc<self.MIN_SEAICE_CONC)] = 0.0  ##discard below threadshold
        return seaice_conc

    def get_seaice_thick(self, **kwargs):
        """
        Get total seaice thickness from the multicategory ice volume (vicen)
        """
        seaice_conc = self.get_seaice_conc(**kwargs)
        
        seaice_volume = np.zeros(self.grid.x.shape)
        rec = kwargs.copy()
        rec['name'] = 'seaice_volume_ncat'
        rec['units'] = self.variables['seaice_volume_ncat']['units']
        for k in range(len(self.variables['seaice_volume_ncat']['levels'])):
            seaice_volume += self.read_var(**{**rec, 'k':k})
        
        seaice_thick = np.zeros(self.grid.x.shape)
        ind = np.where(seaice_conc>=self.MIN_SEAICE_CONC)
        upper_limit = thickness_upper_limit(seaice_conc[ind], 'seaice')
        seaice_thick[ind] = np.minimum(seaice_volume[ind] / seaice_conc[ind], upper_limit)
        return seaice_thick

    def get_snow_thick(self, **kwargs):
        """
        Get total snow thickness from the multi-category snow volume (vsnon)
        """
        seaice_conc = self.get_seaice_conc(**kwargs)

        snow_volume = np.zeros(self.grid.x.shape)
        rec = kwargs.copy()
        rec['name'] = 'snow_volume_ncat'
        rec['units'] = self.variables['snow_volume_ncat']['units']
        for k in range(len(self.variables['snow_volume_ncat']['levels'])):
            snow_volume += self.read_var(**{**rec, 'k':k})
        
        snow_thick = np.zeros(self.grid.x.shape)
        ind = np.where(seaice_conc>=self.MIN_SEAICE_CONC)
        upper_limit = thickness_upper_limit(seaice_conc[ind], 'snow')
        snow_thick[ind] = np.minimum(snow_volume[ind] / seaice_conc[ind], upper_limit)
        return snow_thick

    def preprocess(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        offset = task_id * self.nproc_per_util
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        run_dir = os.path.join(kwargs['path'], mstr[1:], 'SCRATCH')
        ##make sure model run directory exists
        makedir(run_dir)

        ##generate namelists, blkdat, ice_in, etc.
        namelist(self, time, forecast_period, run_dir)

        ##copy synoptic forcing fields from a long record in basedir, will be perturbed later
        for varname in self.force_synoptic_names:
            forcing_file = os.path.join(self.basedir, 'force', 'synoptic', self.E, varname)
            forcing_file_out = os.path.join(run_dir, 'forcing.'+varname)
            f = ABFileForcing(forcing_file, 'r')
            fo = ABFileForcing(forcing_file_out, 'w', idm=f.idm, jdm=f.jdm, cline1=f._cline1, cline2=f._cline2)
            t = time
            dt = self.forcing_dt
            rdtime = dt / 24
            while t <= next_time:
                dtime1 = dayfor(self.yrflag, t.year, int(t.strftime('%j')), t.hour)
                fld = f.read_field(varname, dtime1)
                fo.write_field(fld, None, varname, dtime1, rdtime)
                t += dt * dt1h
            f.close()
            fo.close()

        ##link necessary files for model run
        shell_cmd = f"cd {run_dir}; "
        ##partition setting
        partit_file = os.path.join(self.basedir, 'topo', 'partit', f'depth_{self.R}_{self.T}.{self.nproc:04d}')
        shell_cmd += f"ln -fs {partit_file} patch.input; "
        ##topo files
        for ext in ['.a', '.b']:
            file = os.path.join(self.basedir, 'topo', 'regional.grid'+ext)
            shell_cmd += f"ln -fs {file} .; "
            file = os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}'+ext)
            shell_cmd += f"ln -fs {file} regional.depth{ext}; "
        file = os.path.join(self.basedir, 'topo', f'kmt_{self.R}_{self.T}.nc')
        shell_cmd += f"ln -fs {file} cice_kmt.nc; "
        file = os.path.join(self.basedir, 'topo', 'cice_grid.nc')
        shell_cmd += f"ln -fs {file} .; "
        ##nest files
        nest_dir = os.path.join(self.basedir, 'nest', self.E)
        shell_cmd += f"ln -fs {nest_dir} nest; "
        ##TODO: there is extra logic in nhc_root/bin/expt_preprocess.sh to be added here
        ##relax files
        for ext in ['.a', '.b']:
            for varname in ['intf', 'saln', 'temp']:
                file = os.path.join(self.basedir, 'relax', self.E, 'relax_'+varname[:3]+ext)
                shell_cmd += f"ln -fs {file} {'relax.'+varname+ext}; "
            for varname in ['thkdf4', 'veldf4']:
                file = os.path.join(self.basedir, 'relax', self.E, varname+ext)
                shell_cmd += f"ln -fs {file} {varname+ext}; "
        ##other forcing files
        for ext in ['.a', '.b']:
            ##rivers
            file = os.path.join(self.basedir, 'force', 'rivers', self.E, 'rivers'+ext)
            shell_cmd += f"ln -fs {file} {'forcing.rivers'+ext}; "
            ##seawifs
            file = os.path.join(self.basedir, 'force', 'seawifs', 'kpar'+ext)
            shell_cmd += f"ln -fs {file} {'forcing.kpar'+ext}; "
        run_command(shell_cmd)

        ##copy restart files from restart_dir
        restart_dir = kwargs['restart_dir']
        ##job_submit_cmd = kwargs['job_submit_cmd']
        tstr = time.strftime('%Y_%j_%H_0000')
        for ext in ['.a', '.b']:
            file = os.path.join(restart_dir, 'restart.'+tstr+mstr+ext)
            file1 = os.path.join(kwargs['path'], 'restart.'+tstr+mstr+ext)
            run_command(f"cp -fL {file} {file1}")
            run_command(f"ln -fs {file1} {os.path.join(run_dir, 'restart.'+tstr+ext)}")
        makedir(os.path.join(run_dir, 'cice'))
        tstr = time.strftime('%Y-%m-%d-00000')
        file = os.path.join(restart_dir, 'iced.'+tstr+mstr+'.nc')
        file1 = os.path.join(kwargs['path'], 'iced.'+tstr+mstr+'.nc')
        run_command(f"cp -fL {file} {file1}")
        run_command(f"ln -fs {file1} {os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc')}")
        run_command(f"echo {os.path.join('.', 'cice', 'iced.'+tstr+'.nc')} > {os.path.join(run_dir, 'cice', 'ice.restart_file')}")

    def postprocess(self, task_id=0, **kwargs):
        """Post processing the restart variables for next forecast"""
        ## routines adapted from the EnKF-MPI-TOPAZ/Tools/fixhycom.F90 code
        kwargs = super().parse_kwargs(**kwargs)

        ##adjust ocean layer thickness dp
        rec = kwargs.copy()
        rec['name'] = 'ocean_layer_thick'
        rec['units'] = self.variables['ocean_layer_thick']['units'] ##should be Pa
        levels = list(self.variables['ocean_layer_thick']['levels'])
        dp = np.zeros((len(levels), self.grid.ny, self.grid.nx))
        for ilev, k in enumerate(levels):
            dp[ilev, ...] = self.read_var(**{**rec, 'k':k})
        dp = adjust_dp(dp, -self.depth, self.ONEM)
        for ilev, k in enumerate(levels):
            self.write_var(dp[ilev,...], **{**rec, 'k':k})

        ##loop over fields in restart file
        restart_file = self.filename(**{**kwargs, 'name':'ocean_temp'})
        f = ABFileRestart(restart_file, 'r+', idm=self.grid.nx, jdm=self.grid.ny, mask=True)
        for i, rec in f.fields.items():
            name = rec['field']
            tlevel = rec['tlevel']
            k = rec['k']
            fld = f.read_field(name, tlevel=tlevel, level=k)

            ##reset variables out of their normal range
            if name == 'temp':
                saln = f.read_field('saln', tlevel=tlevel, level=k)
                temp_min = -0.057 * saln
                ind = np.where(fld < temp_min)
                fld[ind] = temp_min[ind]
                fld[np.where(fld > self.MAX_OCEAN_TEMP)] = self.MAX_OCEAN_TEMP
            elif name == 'saln':
                fld[np.where(fld > self.MAX_OCEAN_SALN)] = self.MAX_OCEAN_SALN
                fld[np.where(fld < self.MIN_OCEAN_SALN)] = self.MIN_OCEAN_SALN
            elif name == 'dp':
                fld = dp[levels.index(k), ...]  ##set dp to the adjusted value

            ##write the field back
            f.overwrite_field(fld, None, name, tlevel=tlevel, level=k)
        f.close()

        ##TODO: other postprocessing in fixhycom to be added (ice variables, etc.)

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        member = kwargs['member']
        if member is not None:
            mstr = '_mem{:03d}'.format(member+1)
        else:
            mstr = ''
        run_dir = os.path.join(kwargs['path'], mstr[1:], 'SCRATCH')
        makedir(run_dir)
        log_file = os.path.join(run_dir, "run.log")
        run_command("touch "+log_file)

        ##check if input file exists
        input_files = []
        tstr = time.strftime('%Y_%j_%H_0000')
        for ext in ['.a', '.b']:
            input_files.append(os.path.join(run_dir, 'restart.'+tstr+ext))
        tstr = time.strftime('%Y-%m-%d-00000')
        input_files.append(os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc'))
        for file in input_files:
            if not os.path.exists(file):
                raise RuntimeError(f"topaz.v5.model.run: input file missing: {file}")

        ##early exit if the run is already finished
        if find_keyword_in_file(log_file, 'Exiting hycom_cice'):
            return

        ##build the shell command line
        model_exe = os.path.join(self.basedir, f'expt_{self.X}', 'build', f'src_{self.V}ZA-07Tsig0-i-sm-sse_relo_mpi', 'hycom_cice')
        shell_cmd =  "source "+self.model_env+"; "  ##enter topaz5 env
        shell_cmd += "cd "+run_dir+"; "             ##enter run directory
        shell_cmd += 'JOB_EXECUTE '+model_exe+" >& run.log"
        run_job(shell_cmd, job_name='topaz5', run_dir=run_dir,
                nproc=self.nproc, offset=task_id*self.nproc_per_run,
                walltime=self.walltime, **kwargs)

        ##check output
        watch_log(log_file, 'Exiting hycom_cice')

        ##move the output restart files to forecast_dir
        tstr = next_time.strftime('%Y_%j_%H_0000')
        for ext in ['.a', '.b']:
            file1 = os.path.join(run_dir, 'restart.'+tstr+ext)
            file2 = os.path.join(kwargs['path'], 'restart.'+tstr+mstr+ext)
            run_command(f"mv {file1} {file2}")
        tstr = next_time.strftime('%Y-%m-%d-00000')
        file1 = os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc')
        file2 = os.path.join(kwargs['path'], 'iced.'+tstr+mstr+'.nc')
        run_command(f"mv {file1} {file2}")

    def run_batch(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        assert kwargs['use_job_array'], "use_job_array shall be True if running ensemble in batch mode."

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        nens = kwargs['nens']
        for member in range(nens):
            mstr = '_mem{:03d}'.format(member+1)
            run_dir = os.path.join(kwargs['path'], mstr[1:], 'SCRATCH')
            makedir(run_dir)
            log_file = os.path.join(run_dir, "run.log")
            run_command("touch "+log_file)

            ##check if input file exists
            input_files = []
            tstr = time.strftime('%Y_%j_%H_0000')
            for ext in ['.a', '.b']:
                input_files.append(os.path.join(run_dir, 'restart.'+tstr+ext))
            tstr = time.strftime('%Y-%m-%d-00000')
            input_files.append(os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc'))
            for file in input_files:
                if not os.path.exists(file):
                    raise RuntimeError(f"topaz.v5.model.run: input file missing: {file}")

        ##build the shell command line
        model_exe = os.path.join(self.basedir, f'expt_{self.X}', 'build', f'src_{self.V}ZA-07Tsig0-i-sm-sse_relo_mpi', 'hycom_cice')
        shell_cmd =  "source "+self.model_env+"; "
        shell_cmd += "cd mem$(printf '%03d' JOB_ARRAY_INDEX); " 
        shell_cmd += 'JOB_EXECUTE '+model_exe+" >& run.log"
        run_job(shell_cmd, job_name='topaz5', run_dir=run_dir, array_size=nens,
                nproc=self.nproc, walltime=self.walltime, **kwargs)

        ##check output
        for member in range(nens):
            
            watch_log(log_file, 'Exiting hycom_cice')

            ##move the output restart files to forecast_dir
            tstr = next_time.strftime('%Y_%j_%H_0000')
            for ext in ['.a', '.b']:
                file1 = os.path.join(run_dir, 'restart.'+tstr+ext)
                file2 = os.path.join(kwargs['path'], 'restart.'+tstr+mstr+ext)
                run_command(f"mv {file1} {file2}")
            tstr = next_time.strftime('%Y-%m-%d-00000')
            file1 = os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc')
            file2 = os.path.join(kwargs['path'], 'iced.'+tstr+mstr+'.nc')
            run_command(f"mv {file1} {file2}")

