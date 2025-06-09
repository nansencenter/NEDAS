import os
import glob
from typing import Optional, Any
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import numpy as np

from NEDAS.utils.conversion import units_convert, t2s, dt1h
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.utils.shell_utils import run_command, run_job, makedir
from NEDAS.utils.progress import watch_log, find_keyword_in_file, watch_files
from NEDAS.models import Model

from ..time_format import dayfor
from ..abfile import ABFileRestart, ABFileArchv, ABFileForcing
from ..model_grid import get_topaz_grid, get_depth, get_mean_ssh, stagger, destagger
from .namelist import namelist
from .postproc import adjust_dp, stmt_fns_sigma, stmt_fns_kappaf
from .cice_utils import thickness_upper_limit, adjust_ice_variables, fix_zsin_profile

class Topaz5Model(Model):
    """
    TOPAZ5 model class.
    """
    nhc_root: str
    basedir: str
    model_env: Optional[str]
    reanalysis_code: str
    V: str
    X: str
    T: str
    R: str
    E: str
    idm: int
    jdm: int
    kdm: int
    nproc: int
    nproc_per_run: int
    nproc_per_util: int
    use_job_array: bool
    walltime: int
    stagnant_log_timeout: int
    ens_run_type: str
    meanssh_file: str
    forcing_file: str
    restart_dt: int
    output_dt: int
    forcing_dt: int
    z_units: str
    thflag: int
    thref: float
    thbase: float
    kapref: int
    yrflag: int
    ONEM: float
    MIN_SEAICE_CONC: float
    MAX_OCEAN_TEMP: float
    MIN_OCEAN_SALN: float
    MAX_OCEAN_SALN: float
    Nilayer: int
    saltmax: float
    min_salin: float
    depressT: float
    nsal: float
    msal: float
    aice_thresh: float
    fice_thresh: float
    hice_impact: float

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
            'ocean_saln_daily': {'name':'salin', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':levels, 'units':'psu'},
            'ocean_mixl_depth_daily': {'name':'mix_dpth', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'Pa'},
            'ocean_dense_daily': {'name':'dense', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'?'},
            'ocean_surf_height_daily': {'name':'srfhgt', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'m'},
            }

        self.iced_variables = {
            'seaice_velocity': {'name':('uvel', 'vvel'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m/s'},
            'seaice_conc_ncat':   {'name':'aicen', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_ncat, 'units':1},
            'seaice_volume_ncat':  {'name':'vicen', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_ncat, 'units':'m'},
            'snow_volume_ncat':  {'name':'vsnon', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_ncat, 'units':'m'},
            }

        self.iceh_variables = {
            'seaice_velocity_daily': {'name':('uvel_d', 'vvel_d'), 'dtype':'float', 'is_vector':True, 'dt':self.output_dt, 'levels':level_sfc, 'units':'m/s'},
            'seaice_conc_daily': {'name':'aice_d', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':1},
            'seaice_thick_daily': {'name':'hi_d', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'m'},
            'seaice_surf_temp_daily': {'name':'Tsfc_d', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'C'},
            'seaice_saln_daily': {'name':'sice_d', 'dtype':'float', 'is_vector':False, 'dt':self.output_dt, 'levels':level_sfc, 'units':'psu'},
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
            'atmos_surf_vapor_mix': {'name':'vapmix', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'kg/kg'},
            }
        self.force_synoptic_names = [name for r in self.atmos_forcing_variables.values() for name in (r['name'] if isinstance(r['name'], tuple) else [r['name']])]

        self.diag_variables = {
            'ocean_surf_height': {'name':'ssh', 'operator':self.get_ocean_surf_height, 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m'},
            'ocean_surf_height_anomaly': {'name':'sla', 'operator':self.get_ocean_surf_height_anomaly, 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m'},
            'ocean_surf_temp': {'name':'sst', 'operator':self.get_ocean_surf_temp, 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'C'},
            'seaice_conc': {'name':'sic', 'operator':self.get_seaice_conc, 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':1},
            'seaice_thick': {'name':'sit', 'operator':self.get_seaice_thick, 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m'},
            'snow_thick': {'name':'snwt', 'operator':self.get_snow_thick, 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m'},
            }

        self.variables = {**self.restart_variables,
                          **self.iced_variables,
                          **self.iceh_variables,
                          **self.atmos_forcing_variables,
                          **self.diag_variables,
                          **self.archive_variables}

        self.grid = None
        grid_info_file = os.path.join(self.basedir, 'topo', 'grid.info')
        if self.basedir and os.path.exists(grid_info_file):
            self.grid = get_topaz_grid(grid_info_file)

        self.depthfile = os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}.a')
        if self.grid and self.depthfile and os.path.exists(self.depthfile):
            self.depth, self.grid.mask = get_depth(self.depthfile, self.grid)

        self.meanssh = None
        if self.meanssh_file and os.path.exists(self.meanssh_file):
            self.meanssh = get_mean_ssh(self.meanssh_file, self.grid)

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        ##filename for each model component
        if kwargs['name'] in self.restart_variables:
            tstr = kwargs['time'].strftime('%Y_%j_%H_%M%S')
            file = 'restart.'+tstr+mstr+'.a'

        elif kwargs['name'] in self.iced_variables:
            t = kwargs['time']
            tstr = f"{t:%Y-%m-%d}-{t.hour*3600:05}"
            file = 'iced.'+tstr+mstr+'.nc'

        elif kwargs['name'] in self.iceh_variables:
            tstr = kwargs['time'].strftime('%Y-%m-%d')
            file = os.path.join(mstr[1:], 'SCRATCH', 'cice', 'iceh.'+tstr+'.nc')

        elif kwargs['name'] in self.atmos_forcing_variables:
            file = os.path.join(mstr[1:], 'SCRATCH', 'forcing')
            return os.path.join(kwargs['path'], file)

        elif kwargs['name'] in self.archive_variables:
            tstr = kwargs['time'].strftime('%Y_%j_??')  ##archive variables are daily means
            file = os.path.join(mstr[1:], 'SCRATCH', 'archm.'+tstr+'.a')

        elif kwargs['name'] in self.diag_variables:
            kstr = f"_k{kwargs['k']}_"
            tstr = t2s(kwargs['time'])
            file = os.path.join(mstr[1:], 'SCRATCH', kwargs['name']+kstr+tstr+'.npy')
            return os.path.join(kwargs['path'], file)

        else:
            raise ValueError(f"filename: ERROR: unknown variable name '{kwargs['name']}'")

        path = kwargs['path']
        fname = os.path.join(path, file)
        files = glob.glob(fname)
        if len(files) > 0:
            return files[0]

        ##if no corresponding files found under the given path
        ##try to find two layers above for the other cycle time
        dirs = path.split(os.sep)
        if dirs[-1] == 'topaz.v5' and len(dirs[-2])==12:
            root = os.sep.join(dirs[:-2])
            for tdir in os.listdir(root):
                fname = os.path.join(root, tdir, 'topaz.v5', file)
                files = glob.glob(fname)
                if len(files) > 0:
                    return files[0]
        raise FileNotFoundError(f"filename: ERROR: could not find {file} in {path} or its parent directories")

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var(self, **kwargs):
        if self.grid is None:
            raise AttributeError("topaz5model: grid not yet defined")
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        ##get the variable from restart files
        if name in self.restart_variables:
            f = ABFileRestart(fname, 'r', idm=self.grid.nx, jdm=self.grid.ny)
            if rec['is_vector']:
                var1 = f.read_field(rec['name'][0], level=kwargs['k'], tlevel=1, mask=self.grid.mask)
                var2 = f.read_field(rec['name'][1], level=kwargs['k'], tlevel=1, mask=self.grid.mask)
                var = np.array([var1, var2])
            else:
                var = f.read_field(rec['name'], level=kwargs['k'], tlevel=1, mask=self.grid.mask)
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
            dtime = (kwargs['time'] - datetime(1900,12,31,tzinfo=timezone.utc)) / (24*dt1h)
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
            ## if the npy file exists, one could just read it to get the variable.
            ## but here we always calculate the variable from the model state, and refresh to the npy file, to be safe
            var = rec['operator'](**kwargs)
            np.save(fname, var)

        elif name in self.archive_variables:
            f = ABFileArchv(fname, 'r', mask=True)
            if rec['is_vector']:
                var1 = f.read_field(rec['name'][0], level=kwargs['k'])
                var2 = f.read_field(rec['name'][1], level=kwargs['k'])
                var = np.array([var1, var2])
            else:
                var = f.read_field(rec['name'], level=kwargs['k'])
            f.close()

        else:
            raise ValueError(f"read_var: ERROR: unknown variable name '{name}'")

        ##convert units if necessary
        var = units_convert(rec['units'], kwargs['units'], var)
        return var

    def write_var(self, var, **kwargs):
        if self.grid is None:
            raise AttributeError("topaz5model: grid not yet defined")
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        ##convert back to old units
        var = units_convert(kwargs['units'], rec['units'], var)

        if name in self.restart_variables:
            ##open the restart file for over-writing
            ##the 'r+' mode and a new overwrite_field method were added in the ABFileRestart in .abfile
            f = ABFileRestart(fname, 'r+', idm=self.grid.nx, jdm=self.grid.ny, mask=True)
            if rec['is_vector']:
                for i in range(2):
                    f.overwrite_field(var[i,...], self.grid.mask, rec['name'][i], level=kwargs['k'], tlevel=1)
            else:
                f.overwrite_field(var, self.grid.mask, rec['name'], level=kwargs['k'], tlevel=1)
            f.close()

        elif name in self.iced_variables:
            is_ncat = (name[-5:] == '_ncat')  ##if name is a multicategory variable (categories indexed by k)
            if rec['is_vector']:
                for i in range(2):
                    if is_ncat:
                        dims = {'ncat':None, 'nj':self.grid.ny, 'ni':self.grid.nx}
                        recno = {'ncat':kwargs['k']}
                    else:
                        dims = {'nj':self.grid.ny, 'ni':self.grid.nx}
                        recno = None
                    nc_write_var(fname, dims, rec['name'][i], var[i,...], recno=recno, comm=kwargs['comm'])
            else:
                if is_ncat:
                    dims = {'ncat':None, 'nj':self.grid.ny, 'ni':self.grid.nx}
                    recno = {'ncat':kwargs['k']}
                else:
                    dims = {'nj':self.grid.ny, 'ni':self.grid.nx}
                    recno = None
                nc_write_var(fname, dims, rec['name'], var, recno=recno, comm=kwargs['comm'])

        elif name in self.atmos_forcing_variables:
            dtime = (kwargs['time'] - datetime(1900,12,31,tzinfo=timezone.utc)) / timedelta(days=1)
            if rec['is_vector']:
                for i in range(2):
                    f = ABFileForcing(fname+'.'+rec['name'][i], 'r+')
                    f.overwrite_field(var[i,...], None, rec['name'][i], dtime)
                    f.close()
            else:
                f = ABFileForcing(fname+'.'+rec['name'], 'r+')
                f.overwrite_field(var, None, rec['name'], dtime)
                f.close()

        elif name in self.diag_variables:
            np.save(fname, var)

    def z_coords(self, **kwargs):
        return self._z_coords_cached(**kwargs)

    @lru_cache(maxsize=3)
    def _z_coords_cached(self, **kwargs):
        """
        Calculate vertical coordinates given the 3D model state
        Return:
        - z: np.array
        The corresponding z field
        """
        ##some defaults if not set in kwargs
        if 'k' not in kwargs:
            kwargs['k'] = 0

        z = np.zeros((self.jdm, self.idm))
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
        if self.grid is None:
            raise AttributeError("topaz5model: grid not yet defined")
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
        ssh[self.grid.mask] = np.nan
        return ssh

    def get_ocean_surf_height_anomaly(self, **kwargs):
        if self.grid is None:
            raise AttributeError("topaz5model: grid not yet defined")
        self.meanssh_file = os.path.join(self.basedir, 'topo', 'meanssh.uf')
        assert self.meanssh is not None, f"SLA: cannot find meanssh file {self.meanssh_file}"
        ssh = self.get_ocean_surf_height(**kwargs)
        sla = ssh - self.meanssh
        sla[self.grid.mask] = np.nan
        return sla

    def get_ocean_surf_temp(self, **kwargs):
        #just return first level ocean_temp
        return self.read_var(**{**kwargs, 'name':'ocean_temp', 'k':1})

    def get_seaice_conc(self, **kwargs):
        """
        Get total seaice concentration from multicategory ice concentration (aicen)
        adapted from ReanalysisTP5/SSHFromState_HYCOMICE/mod_read_icednc by J. Xie
        """
        if self.grid is None:
            raise AttributeError("topaz5model: grid not yet defined")
        seaice_conc = np.zeros(self.grid.x.shape)
        rec = kwargs.copy()
        rec['name'] = 'seaice_conc_ncat'
        rec['units'] = self.variables['seaice_conc_ncat']['units']
        for k in range(len(self.variables['seaice_conc_ncat']['levels'])):
            seaice_conc += self.read_var(**{**rec, 'k':k})
        
        seaice_conc[np.where(seaice_conc<self.MIN_SEAICE_CONC)] = 0.0  ##discard below threadshold
        seaice_conc[self.grid.mask] = np.nan
        return seaice_conc

    def get_seaice_thick(self, **kwargs):
        """
        Get total seaice thickness from the multicategory ice volume (vicen)
        """
        if self.grid is None:
            raise AttributeError("topaz5model: grid not yet defined before calling get_seaice_thick")
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
        seaice_thick[self.grid.mask] = np.nan
        return seaice_thick

    def get_snow_thick(self, **kwargs):
        """
        Get total snow thickness from the multi-category snow volume (vsnon)
        """
        if self.grid is None:
            raise AttributeError("topaz5model: grid not yet defined before calling get_snow_thick")
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
            forcing_file = self.forcing_file.format(member=kwargs['member']+1, time=time, name=varname)
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
        tstr = time.strftime('%Y_%j_%H_%M%S')
        for ext in ['.a', '.b']:
            file = os.path.join(restart_dir, 'restart.'+tstr+mstr+ext)
            file1 = os.path.join(kwargs['path'], 'restart.'+tstr+mstr+ext)
            run_command(f"cp -fL {file} {file1}")
            run_command(f"ln -fs {file1} {os.path.join(run_dir, 'restart.'+tstr+ext)}")
        makedir(os.path.join(run_dir, 'cice'))
        tstr = f"{time:%Y-%m-%d}-{time.hour*3600:05}"
        file = os.path.join(restart_dir, 'iced.'+tstr+mstr+'.nc')
        file1 = os.path.join(kwargs['path'], 'iced.'+tstr+mstr+'.nc')
        run_command(f"cp -fL {file} {file1}")
        run_command(f"ln -fs {file1} {os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc')}")
        run_command(f"echo {os.path.join('.', 'cice', 'iced.'+tstr+'.nc')} > {os.path.join(run_dir, 'cice', 'ice.restart_file')}")

    def postprocess(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        if self.grid is None:
            raise AttributeError("topaz5model: grid not yet defined")
        time = kwargs['time']
        member = kwargs['member']
        if member is not None:
            mstr = '_mem{:03d}'.format(member+1)
        else:
            mstr = ''
        run_dir = os.path.join(kwargs['path'], mstr[1:], 'SCRATCH')
        makedir(run_dir)

        # link files
        commands = ""
        if self.model_env:
            commands += f". {self.model_env}; "
        commands += f"cd {run_dir}; "
        for ext in ['.a', '.b']:
            file1 = os.path.join(kwargs['restart_dir'], f'restart.{time:%Y_%j_%H_%M%S}{mstr}{ext}')
            file2 = f'forecast{member+1:03}{ext}'
            commands += f"ln -fs {file1} {file2}; "
        file1 = os.path.join(kwargs['restart_dir'], f'iced.{time:%Y-%m-%d}-{time.hour*3600:05}{mstr}.nc')
        file2 = f'ice_forecast{member+1:03}.nc'
        commands += f"ln -fs {file1} {file2}; "
        commands += f"ln -fs {self.reanalysis_code}/FILES/depths{self.idm}x{self.jdm}.uf .; "
        run_command(commands)

        # restart2nc on forecast files
        commands = ""
        if self.model_env:
            commands += f". {self.model_env}; "
        commands += f"cd {run_dir}; "
        commands += f"{os.path.join(self.reanalysis_code, 'ASSIM', 'BIN', 'restart2nc')} forecast{member+1:03}.a ice_forecast{member+1:03}.nc"
        run_command(commands)

        # add posterior ice variables in analysis abfile
        for ext in ['.a', '.b']:
            file1 = os.path.join(kwargs['path'], f'restart.{time:%Y_%j_%H_%M%S}{mstr}{ext}')
            file2 = os.path.join(run_dir, f'analysis{member+1:03}{ext}')
            run_command(f"cp -L {file1} {file2}")
        f = ABFileRestart(file2, 'r+', idm=self.grid.nx, jdm=self.grid.ny, mask=True)
        nfld = len(f.fields.keys())
        fld = np.load(self.filename(path=kwargs['path'], name='seaice_conc', member=member, time=time))
        f.write_field(fld, None, 'ficem', 0, 1, nfld)
        fld = np.load(self.filename(path=kwargs['path'], name='seaice_thick', member=member, time=time))
        f.write_field(fld, None, 'hicem', 0, 1, nfld+1)
        f.close()

        # run fixhycom and update restart file
        commands = ""
        if self.model_env:
            commands += f". {self.model_env}; "
        commands += f"cd {run_dir}; "
        commands += f"{os.path.join(self.reanalysis_code, 'ASSIM', 'BIN', 'fixhycom')} analysis{member+1:03}.a {member+1} forecast{member+1:03}.nc ice_forecast{member+1:03}.nc {time:%j} 0; "
        commands += f"cat fixanalysis{member+1:03}.b >> tmp{member+1:03}.b; mv tmp{member+1:03}.b fixanalysis{member+1:03}.b; "
        for ext in ['.a', '.b']:
            file1 = os.path.join(run_dir, f'fixanalysis{member+1:03}{ext}')
            file2 = os.path.join(kwargs['path'], f'restart.{time:%Y_%j_%H_%M%S}{mstr}{ext}')
            commands += f"mv {file1} {file2}; "
        file1 = os.path.join(run_dir, f'fix_ice_forecast{member+1:03}.nc')
        file2 = os.path.join(kwargs['path'], f'iced.{time:%Y-%m-%d}-{time.hour*3600:05}{mstr}.nc')
        commands += f"mv {file1} {file2}; "
        run_command(commands)

    def postprocess_native(self, task_id=0, **kwargs):
        """Post processing the restart variables for next forecast"""
        ## routines adapted from the EnKF-MPI-TOPAZ/Tools/fixhycom.F90 code
        kwargs = super().parse_kwargs(**kwargs)
        if self.grid is None:
            raise AttributeError("topaz5model: grid not yet defined")

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

        ##fix sea ice variables, from enkf-topaz/Tools/m_put_mod_fld_nc: fix_cice
        restart_dir = kwargs['restart_dir']
        prior_ice_file = self.filename(**{**kwargs, 'path':restart_dir, 'name':'seaice_conc_ncat'})
        post_ice_file = self.filename(**{**kwargs, 'name':'seaice_conc_ncat'})
        fice = self.read_var(**{**kwargs, 'name':'seaice_conc', 'k':0, 'units':1})
        hice = self.read_var(**{**kwargs, 'name':'seaice_thick', 'k':0, 'units':'m'})
        zSin, Tmlt = fix_zsin_profile(self.Nilayer+1, self.saltmax, self.depressT, self.nsal, self.msal)
        adjust_ice_variables(prior_ice_file, post_ice_file, fice, hice, self.grid.mask,
                             self.aice_thresh, self.fice_thresh, self.hice_impact, zSin, Tmlt)

        ##update the diagnostic ice variables
        self.write_var(fice, **{**kwargs, 'name':'seaice_conc', 'k':0, 'units':1})
        self.write_var(hice, **{**kwargs, 'name':'seaice_thick', 'k':0, 'units':'m'})

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

        run_success = False
        ##early exit if the run is already finished
        if find_keyword_in_file(log_file, 'Exiting hycom_cice'):
            run_success = True

        else:
            ##check if input file exists
            input_files = []
            tstr = time.strftime('%Y_%j_%H_%M%S')
            for ext in ['.a', '.b']:
                input_files.append(os.path.join(run_dir, 'restart.'+tstr+ext))
            tstr = f"{time:%Y-%m-%d}-{time.hour*3600:05}"
            input_files.append(os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc'))
            for file in input_files:
                if not os.path.exists(file):
                    raise RuntimeError(f"topaz.v5.model.run: input file missing: {file}")

            ##clean up some files from previous runs
            run_command(f"cd {run_dir}; rm -f archm.* ovrtn_out summary_out")
            run_command(f"echo > {log_file}")

            ##build the shell command line
            model_exe = os.path.join(self.basedir, f'expt_{self.X}', 'build', f'src_{self.V}ZA-07Tsig0-i-sm-sse_relo_mpi', 'hycom_cice')
            shell_cmd = ""
            if self.model_env:
                shell_cmd =  ". "+self.model_env+"; "  ##enter topaz5 env
            shell_cmd += "cd "+run_dir+"; "             ##enter run directory
            shell_cmd += 'JOB_EXECUTE '+model_exe+" >& run.log"

            ##run the model, give it 3 attempts
            for i in range(3):
                try:
                    run_job(shell_cmd, job_name='topaz5', run_dir=run_dir,
                            nproc=self.nproc, offset=task_id*self.nproc_per_run,
                            walltime=self.walltime, log_file=log_file, **kwargs)
                except RuntimeError as e:
                    print(f"{e}, retrying ({2-i} attempts remain)")
                    run_command(f"cp {log_file} {log_file}.attempt{i}")
                    continue
                ##check output
                if find_keyword_in_file(log_file, 'Exiting hycom_cice'):
                    run_success = True
                    break
        assert run_success, f"model run failed after 3 attempts, check in {run_dir}"

        ##move the output restart files to forecast_dir
        tstr = next_time.strftime('%Y_%j_%H_%M%S')
        for ext in ['.a', '.b']:
            file1 = os.path.join(run_dir, 'restart.'+tstr+ext)
            file2 = os.path.join(kwargs['path'], 'restart.'+tstr+mstr+ext)
            if os.path.exists(file1):
                run_command(f"mv {file1} {file2}")
            else:
                assert os.path.exists(file2), f"error moving output file {file1} to {file2}"
        tstr = f"{next_time:%Y-%m-%d}-{next_time.hour*3600:05}"
        file1 = os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc')
        file2 = os.path.join(kwargs['path'], 'iced.'+tstr+mstr+'.nc')
        if os.path.exists(file1):
            run_command(f"mv {file1} {file2}")
        else:
            assert os.path.exists(file2), f"error moving output file {file1} to {file2}"

    def run_batch(self, **kwargs):
        raise NotImplementedError("topaz5model.run_batch: not implemented, use run() instead")
