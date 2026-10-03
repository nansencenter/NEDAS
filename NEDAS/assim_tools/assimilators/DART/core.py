import os
import copy
import ctypes
from datetime import datetime
import numpy as np
from NEDAS.core import Context, Assimilator
from NEDAS.assim_tools.assimilators.serial import SerialAssimilator

# generic DART quantities/obs types compiled into the library (obs_def_nedas_mod.f90)
NUM_SLOTS = 40
DART_EPOCH = datetime(1601, 1, 1)
_f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
_i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
_CALLBACK = ctypes.CFUNCTYPE(None)
_LIB = None  # module level: a CDLL does not pickle
DEFAULT_INFLATION = {
    'prior': {'flavor': 0, 'initial': 1.0, 'sd_initial': 0.6, 'damping': 0.9, 'lower_bound': 1.0,
              'upper_bound': 1e6, 'sd_lower_bound': 0.6, 'sd_max_change': 1.05, 'deterministic': True},
    'posterior': {'flavor': 0, 'initial': 1.0, 'sd_initial': 0.0, 'damping': 1.0, 'lower_bound': 1.0,
                  'upper_bound': 1e6, 'sd_lower_bound': 0.0, 'sd_max_change': 1.05, 'deterministic': True},
}

def default_lib_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'libdartfilter.so')

def load_dart_filter(lib_path: str) -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"{lib_path} not found; build it with build_dart_filter.sh")
        lib = ctypes.CDLL(lib_path)
        p = ctypes.c_void_p
        lib.dart_filter_set_state.argtypes = [ctypes.c_int]*3 + [p, _f64, _f64, _f64, _i32, p] + [ctypes.c_int]*2
        lib.dart_filter_set_obs.argtypes = [ctypes.c_int] + [p]*9
        lib.dart_filter_set_periodic.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double]*2
        lib.dart_filter_set_posterior.argtypes = [p, _CALLBACK]
        lib.dart_filter_set_write_obs_seq.argtypes = [ctypes.c_int]
        lib.dart_filter_run.argtypes = [ctypes.c_int]
        for f in ('dart_filter_set_state', 'dart_filter_set_obs', 'dart_filter_set_posterior', 'dart_filter_set_periodic',
                  'dart_filter_set_write_obs_seq', 'dart_filter_run'):
            getattr(lib, f).restype = None
        _LIB = lib
    return _LIB

def nml_value(v) -> str:
    if isinstance(v, bool):
        return '.true.' if v else '.false.'
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, (list, tuple)):
        return ', '.join(nml_value(x) for x in v)
    return repr(float(v)) if isinstance(v, float) else str(v)

def dart_time(t: datetime) -> tuple[int, int]:
    dt = t.replace(tzinfo=None) - DART_EPOCH   # NEDAS times are UTC
    return dt.days, dt.seconds


class DARTAssimilator(Assimilator):
    """
    DART's own filter_main (libdartfilter.so), with its file I/O replaced by NEDAS memory.

    NEDAS does the state/obs preparation, the transpose to ensemble-complete blocks and H(x);
    DART runs inflation, QC, QCEFF and the serial filter_assim on NEDAS's communicator.
    See build_dart_filter.sh and nedas_hooks_mod.f90.
    """
    assim_mode = 'batch'   # obs_post is recomputed from the posterior by the scheme

    # NEDAS's serial partitioning: one strided block per rank
    init_partitions = SerialAssimilator.init_partitions
    assign_obs = SerialAssimilator.assign_obs
    distribute_partitions = SerialAssimilator.distribute_partitions

    def __init__(self, c: Context):
        defaults = copy.deepcopy(DEFAULT_INFLATION)
        super().__init__(c)
        for f in defaults:   # a partial inflation entry in the config keeps the other defaults
            defaults[f].update((self.inflation or {}).get(f) or {})
        self.inflation = defaults
        self.lib_path = self.dart_lib or default_lib_path()

    # ---------------------------------------------------------------- checks

    def check(self, c: Context) -> None:
        if c.state.info.scalars:
            raise NotImplementedError("DART: scalar state variables are not supported")
        inf = c.inflation_func
        if (inf.prior or inf.post) and (getattr(inf, 'adaptive', False) or getattr(inf, 'coef', 1.0) != 1.0):
            raise NotImplementedError("DART: use assimilator_def.inflation, not inflation_def "
                                      "(DART inflates inside filter)")
        if getattr(c.grid, 'distance_type', 'cartesian') != 'cartesian':
            raise NotImplementedError("DART: only cartesian grids (threed_cartesian location)")
        for rec in c.obs.info.records.values():
            if np.isfinite(rec.troi):
                raise NotImplementedError("DART: DART has no temporal localization, set troi: inf")
            if any(f != 1 for f in rec.impact_on_variable):
                raise NotImplementedError("DART: impact_on_variable is not supported")
        if self.inflation['posterior']['flavor'] not in (0, 4) and not self.compute_posterior(c):
            raise RuntimeError("posterior adaptive inflation needs compute_posterior")

    def compute_posterior(self, c: Context) -> bool:
        return self.inflation['posterior']['flavor'] in (2, 3, 5)

    # ---------------------------------------------------------------- slots

    def slots(self, c: Context) -> dict:
        """DART quantity/type slot (1-based) for each (variable name, component)"""
        keys = set()
        for rec in c.state.info.fields.values():
            keys.update([(rec.name, v) for v in ((0, 1) if rec.is_vector else (-1,))])
        for rec in c.obs.info.records.values():
            keys.update([(rec.name, v) for v in ((0, 1) if rec.is_vector else (-1,))])
        keys = sorted(keys)
        if len(keys) > NUM_SLOTS:
            raise NotImplementedError(f"DART: more than {NUM_SLOTS} variables")
        return {k: i+1 for i, k in enumerate(keys)}

    def slot_name(self, key) -> str:
        name, v = key
        return name if v < 0 else f"{name}_{'xy'[v]}"

    # ---------------------------------------------------------------- transposes

    def transpose_to_ensemble_complete(self, c: Context) -> None:
        # only the state: DART distributes the obs itself
        c.state.state_prior = c.logger('Transpose prior state')(c.state.transpose_to_ensemble_complete)(c, c.state.fields_prior, c.mem_list)
        c.state.state_z = c.logger('Transpose z coordinates')(c.state.transpose_to_ensemble_complete)(c, c.state.fields_z, c.mem_list)

    # ---------------------------------------------------------------- main

    def assimilation_algorithm(self, c: Context) -> None:
        self.check(c)
        lib = load_dart_filter(self.lib_path)
        slots = self.slots(c)

        c.state.state_post = copy.deepcopy(c.state.state_prior)
        par_id = c.pid_mem
        data = c.state.pack_local_state_data(c, par_id, c.state.state_prior, c.state.state_z, c.state.state_static)
        nfld, nloc = data['state_prior'].shape[1:]
        nk = nfld * nloc
        state = np.ascontiguousarray(data['state_prior'].reshape(c.nens, nk))
        if np.isnan(state).any():
            raise ValueError("DART: NaN in the state")
        comp = [v if v is not None else -1 for _, v in data['field_ids']]
        names = [c.state.info.fields[r].name for r, _ in data['field_ids']]
        qty = np.repeat([slots[(n, v)] for n, v in zip(names, comp)], nloc).astype(np.int32)
        x = np.ascontiguousarray(np.tile(data['x'], nfld), dtype=np.float64)
        y = np.ascontiguousarray(np.tile(data['y'], nfld), dtype=np.float64)
        z = np.ascontiguousarray(data['z'].ravel(), dtype=np.float64)
        nmax = c.comm.allreduce(nk, op=c.comm._MPI.MAX) if c.comm.mpi_ready else nk

        workdir = os.path.join(c.fs.analysis_dir(c.time, c.iter), 'dart')
        inf, from_restart = self.load_inflation(c, nk)
        obs = self.collect_obs(c, slots)

        if c.pid == 0:
            os.makedirs(workdir, exist_ok=True)
            self.write_input_nml(c, workdir, slots, obs, from_restart)
        c.comm.Barrier()

        lib.dart_filter_set_periodic(*self.periodic(c))
        days, secs = dart_time(c.time)
        lib.dart_filter_set_state(c.nens, nk, nmax, state.ctypes.data, x, y, z, qty,
                                  inf.ctypes.data, days, secs)
        ptr = lambda a: a.ctypes.data
        lib.dart_filter_set_obs(len(obs['val']), *(ptr(obs[k]) for k in
                                ('type', 'x', 'y', 'z', 'days', 'secs', 'val', 'errvar', 'prior')))
        post = np.full_like(obs['prior'], np.nan)
        callback = _CALLBACK(lambda: self.posterior_obs(c, par_id, data, state, obs, post))
        lib.dart_filter_set_posterior(post.ctypes.data, callback)
        lib.dart_filter_set_write_obs_seq(int(self.write_obs_seq_final))

        cwd = os.getcwd()
        try:
            os.chdir(workdir)
            lib.dart_filter_run(self.fortran_comm(c))
        finally:
            os.chdir(cwd)

        data['state_prior'][:] = state.reshape(c.nens, nfld, nloc)
        c.state.unpack_local_state_data(c, par_id, c.state.state_post, data)
        np.save(os.path.join(workdir, f'inflation.{c.pid}.npy'), inf)

    @staticmethod
    def fortran_comm(c: Context) -> int:
        if c.comm.mpi_ready:
            return c.comm._comm.py2f()
        from mpi4py import MPI
        return MPI.COMM_WORLD.py2f()

    # ---------------------------------------------------------------- obs

    def collect_obs(self, c: Context, slots: dict) -> dict:
        """Every obs with its H(x) ensemble, on all ranks, sorted by time (DART needs that)"""
        seqs, priors = {}, {}
        for part in c.comm_rec.allgather(c.obs.obs_seq):
            seqs.update(part)
        for part in c.comm.allgather(c.obs.obs_prior):
            priors.update(part)

        cols = {k: [] for k in ('type', 'x', 'y', 'z', 't', 'val', 'errvar', 'prior', 'index')}
        for rec_id in sorted(seqs):
            rec = c.obs.info.records[rec_id]
            seq = seqs[rec_id]
            n = seq['obs'].shape[-1]
            ncomp = 2 if rec.is_vector else 1
            comps = (0, 1) if rec.is_vector else (-1,)
            prior = np.stack([np.reshape(priors[m, rec_id], (ncomp, n)) for m in range(c.nens)])
            cols['val'].append(np.reshape(seq['obs'], (ncomp, n)).T.ravel())
            cols['prior'].append(prior.transpose(0, 2, 1).reshape(c.nens, n*ncomp))
            cols['type'].append(np.tile([slots[(rec.name, v)] for v in comps], n))
            for k in ('x', 'y', 'z'):
                cols[k].append(np.repeat(np.asarray(seq[k], dtype=np.float64), ncomp))
            cols['errvar'].append(np.repeat(np.asarray(seq['err_std'], dtype=np.float64)**2, ncomp))
            cols['t'].append(np.repeat(np.asarray(seq['t']), ncomp))
            cols['index'] += [(rec_id, i, v) for i in range(n) for v in comps]

        if not cols['val']:
            empty = np.zeros(0)
            return {'type': empty.astype(np.int32), 'x': empty, 'y': empty, 'z': empty,
                    'days': empty.astype(np.int32), 'secs': empty.astype(np.int32), 'val': empty,
                    'errvar': empty, 'prior': np.zeros((c.nens, 0)), 'index': [], 'hroi': {}, 'vroi': {}}

        obs = {k: np.concatenate(v) for k, v in cols.items() if k not in ('prior', 'index')}
        obs['prior'] = np.concatenate(cols['prior'], axis=1)
        valid = np.isfinite(obs['val']) & np.isfinite(obs['prior']).all(axis=0)
        times = [dart_time(t) for t in obs['t']]
        order = sorted(np.flatnonzero(valid), key=lambda i: times[i])
        out = {k: np.ascontiguousarray(obs[k][order], dtype=np.float64) for k in ('x', 'y', 'z', 'val', 'errvar')}
        out['type'] = np.ascontiguousarray(obs['type'][order], dtype=np.int32)
        out['days'] = np.array([times[i][0] for i in order], dtype=np.int32)
        out['secs'] = np.array([times[i][1] for i in order], dtype=np.int32)
        out['prior'] = np.ascontiguousarray(obs['prior'][:, order])
        out['index'] = [cols['index'][i] for i in order]
        return out

    def posterior_obs(self, c: Context, par_id, data, state, obs, post) -> None:
        """Called from inside filter_main: H(x_post) with NEDAS's own obs operators"""
        data['state_prior'][:] = state.reshape(data['state_prior'].shape)
        c.state.unpack_local_state_data(c, par_id, c.state.state_post, data)
        # the transpose deletes what it sends, and state_post is needed again after filter_main
        c.state.fields_post = c.state.transpose_to_field_complete(c, copy.deepcopy(c.state.state_post))
        c.state.output_state(c, 'post')
        c.obs.prepare_obs_from_state(c, 'post')
        obs_post = {}
        for part in c.comm.allgather(c.obs.obs_post):
            obs_post.update(part)
        for key, (rec_id, i, v) in enumerate(obs['index']):
            for m in range(c.nens):
                seq = obs_post[m, rec_id]
                post[m, key] = seq[v, i] if v >= 0 else seq[i]

    # ---------------------------------------------------------------- inflation

    def load_inflation(self, c: Context, nk: int):
        """Inflation fields from the previous cycle (this rank's block), else namelist values"""
        inf = np.ones((4, nk))
        inf[1::2] = 0.0
        adaptive = {f: self.inflation[f]['flavor'] in (2, 3, 5) for f in ('prior', 'posterior')}
        found = False
        if any(adaptive.values()):
            prev = os.path.join(c.fs.analysis_dir(c.prev_time, c.iter), 'dart', f'inflation.{c.pid}.npy')
            if os.path.exists(prev):
                saved = np.load(prev)
                if saved.shape == inf.shape:
                    inf[:] = saved
                    found = True
        found = c.comm.allreduce(int(found), op=c.comm._MPI.MIN) if c.comm.mpi_ready else int(found)
        return np.ascontiguousarray(inf), {f: bool(found) and a for f, a in adaptive.items()}

    # ---------------------------------------------------------------- namelists

    def cyclic(self, c: Context):
        grid = c.grid
        if hasattr(grid, 'cyclic_dim'):
            cyc = str(grid.cyclic_dim or '')
            return 'x' in cyc, 'y' in cyc
        return bool(getattr(grid, 'cyclic', False)), False

    def periodic(self, c: Context) -> tuple:
        """(x_on, xmin, xmax, y_on, ymin, ymax) for DART's set_periodic"""
        cx, cy = self.cyclic(c)
        g = c.grid
        xr = (float(g.xmin), float(g.xmin + g.Lx)) if cx else (0.0, 0.0)
        yr = (float(g.ymin), float(g.ymin + g.Ly)) if cy else (0.0, 0.0)
        if cy and (not cx or yr != xr):
            # threed_cartesian sizes periodic y boxes with the x limits
            raise NotImplementedError("DART: periodic y needs a periodic x with the same extent")
        return (int(cx), *xr, int(cy), *yr)

    def write_input_nml(self, c: Context, workdir: str, slots: dict, obs: dict, from_restart: dict) -> None:
        used = sorted(slots.values())
        types = [f'NEDAS_{s:02d}' for s in used]
        hroi = {}
        vroi = {}
        for rec in c.obs.info.records.values():
            for v in ((0, 1) if rec.is_vector else (-1,)):
                s = slots[(rec.name, v)]
                if hroi.setdefault(s, rec.hroi) != rec.hroi or vroi.setdefault(s, rec.vroi) != rec.vroi:
                    raise NotImplementedError(f"DART: '{rec.name}' has records with different hroi/vroi")
        big = 1e30
        half = {s: (h/2 if np.isfinite(h) else big) for s, h in hroi.items()}
        vnorm = {s: (vroi[s]/hroi[s] if np.isfinite(vroi[s]) and np.isfinite(hroi[s]) else big) for s in hroi}
        cutoff = max(half.values()) if half else big
        vn = max(vnorm.values()) if vnorm else big
        special = [s for s in half if half[s] != cutoff]
        special_vn = [s for s in vnorm if vnorm[s] != vn]

        inf_keys = ('flavor', 'initial_from_restart', 'sd_initial_from_restart', 'deterministic', 'initial',
                    'sd_initial', 'damping', 'lower_bound', 'upper_bound', 'sd_lower_bound', 'sd_max_change')
        infl = {f: dict(self.inflation[f]) for f in ('prior', 'posterior')}
        for f in infl:
            infl[f]['initial_from_restart'] = infl[f]['sd_initial_from_restart'] = from_restart[f]

        cx, cy = self.cyclic(c)
        nml = {
            'utilities_nml': {'termlevel': 1, 'logfilename': 'dart_log.out', 'nmlfilename': 'dart_log.nml',
                              'write_nml': 'file', 'module_details': False},
            'mpi_utilities_nml': {},
            'ensemble_manager_nml': {'layout': 1},
            'filter_nml': {
                'ens_size': c.nens, 'num_groups': self.num_groups, 'distributed_state': True,
                'init_time_days': -1, 'init_time_seconds': -1,
                'single_file_in': False, 'single_file_out': False, 'perturb_from_single_instance': False,
                'stages_to_write': 'output', 'output_members': True, 'output_mean': False, 'output_sd': False,
                'num_output_state_members': 0, 'num_output_obs_members': 0,
                'compute_posterior': self.compute_posterior(c),
                **{f'inf_{k}': [infl['prior'][k], infl['posterior'][k]] for k in inf_keys},
            },
            'assim_tools_nml': {
                'cutoff': cutoff, 'sort_obs_inc': self.sort_obs_inc, 'spread_restoration': False,
                'sampling_error_correction': self.sampling_error_correction,
                'adaptive_localization_threshold': self.adaptive_localization_threshold,
                'print_every_nth_obs': 0, 'close_obs_caching': True,
                'rectangular_quadrature': self.rectangular_quadrature,
                'gaussian_likelihood_tails': self.gaussian_likelihood_tails,
                **({'special_localization_obs_types': [f'NEDAS_{s:02d}' for s in special],
                    'special_localization_cutoffs': [half[s] for s in special]} if special else {}),
            },
            'cov_cutoff_nml': {'select_localization': 1},
            'reg_factor_nml': {'select_regression': 1},
            'obs_sequence_nml': {'write_binary_obs_sequence': False},
            'obs_kind_nml': {'assimilate_these_obs_types': types, 'use_precomputed_FOs_these_obs_types': types},
            'location_nml': {
                # get_close's box search does not wrap (only get_dist does): one box per periodic axis
                **({'nx': 1} if cx else {}), **({'ny': 1} if cy else {}),
                'vert_normalization_height': vn,
                **({'special_vert_normalization_obs_types': [f'NEDAS_{s:02d}' for s in special_vn],
                    'special_vert_normalization_heights': [vnorm[s] for s in special_vn]} if special_vn else {}),
            },
            'quality_control_nml': {'input_qc_threshold': 3.0, 'outlier_threshold': self.outlier_threshold},
            'state_vector_io_nml': {},
            'algorithm_info_nml': {'qceff_table_filename': self.write_qceff_table(c, workdir, slots)},
            'probit_transform_nml': {},
            'kde_nml': {'quadrature_order': self.quadrature_order},
        }
        for section, entries in (self.namelist or {}).items():
            nml.setdefault(section, {}).update(entries)
        if self.sampling_error_correction:
            dst = os.path.join(workdir, 'sampling_error_correction_table.nc')
            if not os.path.exists(dst):
                os.symlink(self.sec_table, dst)

        lines = ['! written by NEDAS (assim_tools/assimilators/DART)']
        for section, entries in nml.items():
            lines.append(f'&{section}')
            lines += [f'   {k} = {nml_value(v)}' for k, v in entries.items()]
            lines.append('   /\n')
        with open(os.path.join(workdir, 'input.nml'), 'w') as f:
            f.write('\n'.join(lines))

    def write_qceff_table(self, c: Context, workdir: str, slots: dict) -> str:
        """QCEFF table from filter_kind/probit_dist and per-variable qceff entries; '' = all defaults"""
        if self.filter_kind == 'EAKF' and self.probit_dist == 'NORMAL_DISTRIBUTION' and not self.qceff:
            return ''
        b = lambda x: '.true.' if x else '.false.'
        rows = []
        for key, s in sorted(slots.items(), key=lambda kv: kv[1]):
            opt = {'filter_kind': self.filter_kind, 'dist': self.probit_dist, 'bounded_below': False,
                   'bounded_above': False, 'lower_bound': -888888, 'upper_bound': -888888}
            opt.update((self.qceff or {}).get(key[0], {}))
            opt.update((self.qceff or {}).get(self.slot_name(key), {}))
            bounds = f"{b(opt['bounded_below'])},{b(opt['bounded_above'])},{opt['lower_bound']},{opt['upper_bound']}"
            dist = f"{opt['dist']},{bounds}"
            rows.append(f"QTY_NEDAS_{s:02d},{bounds},{dist},{dist},{dist},{opt['filter_kind']},{bounds}")
        head = ('QCEFF table version: 1,obs_error_info,,,,probit_inflation,,,,,probit_state,,,,,'
                'probit_extended_state,,,,,obs_inc_info,,,,\n'
                'QTY_NAME:' + ',bounded_below,bounded_above,lower_bound,upper_bound' +
                (',dist_type,bounded_below,bounded_above,lower_bound,upper_bound' * 3) +
                ',filter_kind,bounded_below,bounded_above,lower_bound,upper_bound\n')
        with open(os.path.join(workdir, 'qceff_table.csv'), 'w') as f:
            f.write(head + '\n'.join(rows) + '\n')
        return 'qceff_table.csv'
