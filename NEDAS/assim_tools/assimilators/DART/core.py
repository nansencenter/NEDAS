import ctypes
import os
import re
import subprocess
import sys
import numpy as np
from numpy.ctypeslib import ndpointer
from NEDAS.assim_tools.assimilators.serial import SerialAssimilator
from NEDAS.utils.conversion import t2h

_f64 = ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')

# must match the KIND_* parameters in dart_kernels.f90
FILTER_KINDS = {'EAKF': 1, 'ENKF': 2, 'KERNEL': 3, 'PARTICLE': 4,
                'RHF': 5, 'GAMMA': 6, 'BNRHF': 7, 'KDE': 8}

# kernels that draw from DART's random sequence, and so must be seeded before use
STOCHASTIC_KINDS = {'ENKF', 'KERNEL'}

# Kernels whose behaviour DART takes from a namelist, so they need DART's utilities up and
# its namelists read before they run:
#   ENKF  sort_obs_inc
#   RHF   rectangular_quadrature, gaussian_likelihood_tails
#   KDE   quadrature_order (kde_nml, read on first use)
#
# The others read no namelist variable that reaches them, so they stay free of DART's
# runtime setup -- which is what lets EAKF reproduce NEDAS's native results exactly.
# KERNEL is deliberately absent: it looked like it belonged here because its seeding block
# calls my_task_id(), but seeding explicitly through dart_set_random_seed() skips that.
KINDS_NEEDING_INIT = {'ENKF', 'RHF', 'KDE'}

# Sections DART insists on when we initialize. utilities_nml is read by
# initialize_utilities, assim_tools_nml by assim_tools_init, and obs_kind_nml by the
# obs_kind module that assim_tools_init pulls in through get_num_types_of_obs(). None of
# them are optional: a missing section makes DART stop the process. kde_nml is read with
# optional_nml so it may be absent, but we write it to carry quadrature_order.
REQUIRED_NML_SECTIONS = ('utilities_nml', 'assim_tools_nml', 'obs_kind_nml')

# first line of an input.nml we wrote, so we never overwrite a real DART namelist
NEDAS_NML_MARKER = '! written by NEDAS (assim_tools/assimilators/DART) -- safe to delete'

# status codes returned by dart_obs_increment
_STATUS = {1: "obs error variance and prior spread are both zero",
           2: "unknown filter_kind",
           3: "likelihood underflowed (BNRHF); check the bounds"}

def load_dart_kernels(lib_path: str) -> ctypes.CDLL:
    """
    Load libdartkernels.so and declare the signatures of the wrappers in dart_kernels.f90
    """
    lib = ctypes.CDLL(lib_path)

    lib.dart_obs_increment.restype = ctypes.c_int
    lib.dart_obs_increment.argtypes = [ctypes.c_int, ctypes.c_int, _f64,
                                       ctypes.c_double, ctypes.c_double,
                                       ctypes.c_int, ctypes.c_int,
                                       ctypes.c_double, ctypes.c_double, _f64, _f64]

    lib.dart_update_from_obs_inc.restype = None
    lib.dart_update_from_obs_inc.argtypes = [ctypes.c_int, ctypes.c_int, _f64, _f64,
                                             ctypes.c_double, _f64, _f64]

    lib.dart_set_random_seed.restype = None
    lib.dart_set_random_seed.argtypes = [ctypes.c_int]

    lib.dart_initialize.restype = None
    lib.dart_initialize.argtypes = []
    return lib

def default_lib_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'libdartkernels.so')

def _fortran_bool(value) -> str:
    return '.true.' if value else '.false.'

def dart_error_message(output: str) -> str:
    """
    Pull DART's complaint out of a captured run.

    DART reports fatal conditions as a block ending in 'message: ...' before stopping, so
    that line is the useful part; fall back to the tail of the output if the format changes.
    """
    match = re.search(r'message:\s*(.+)', output)
    if match:
        return match.group(1).strip()
    return output.strip()[-400:] or '(no output captured)'

class DARTAssimilator(SerialAssimilator):
    """
    Serial filters using DART's own compiled kernels rather than NEDAS's native ports.

    NEDAS keeps the grid, partitioning, localization, obs matching and I/O; only
    obs_increment and the regression onto the state come from DART (via ctypes into
    libdartkernels.so, see dart_kernels.f90 and build_dart_kernels.sh).

    assimilator_def.filter_kind selects the kernel (see default.yml). With filter_kind
    EAKF the results should match NEDAS's native EAKF assimilator to roundoff --
    tests/test_dart_kernels.py checks exactly that, so upstream numerics changes show up
    as a test failure rather than as silent drift. The other kernels have no native NEDAS
    counterpart to compare against.

    DART keeps several kernel options in its namelists rather than in arguments. Those are
    exposed here as assimilator_def entries (sort_obs_inc, rectangular_quadrature,
    gaussian_likelihood_tails, quadrature_order) and written into an input.nml for DART to
    read; see _ensure_initialized(). Only the kernels in KINDS_NEEDING_INIT pay that cost.

    On error handling: DART reports fatal conditions by calling error_handler, which ends
    the process -- there is no exception for python to catch. Everything that can be checked
    beforehand therefore is (see _check_gated_options and _ensure_initialized), and the one
    remaining abort-prone call, initialization, is rehearsed in a subprocess first so its
    failure arrives as a python exception. The per-observation kernel calls are too hot to
    wrap that way and rely on the status codes returned by dart_obs_increment instead.

    Static members (covariance_def.nens_static) are not supported: DART's kernels take the
    dynamic ensemble alone. check_capabilities() rejects that configuration up front, so the
    static arguments below are accepted to satisfy the SerialAssimilator interface and ignored.
    """
    dart_lib: str = ''
    filter_kind: str = 'EAKF'
    random_seed: int = 0          # 0: derive a seed from the analysis time
    write_input_nml: bool = True  # may write an input.nml for the kernels that need one

    # DART namelist options that reach the kernels we call (assim_tools_nml / kde_nml)
    sort_obs_inc: bool = True
    rectangular_quadrature: bool = True
    gaussian_likelihood_tails: bool = False
    quadrature_order: int = 9
    sampling_error_correction: bool = False   # gated, see _check_gated_options

    bounded_below: bool = False
    bounded_above: bool = False
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    _lib = None
    _net_a: float = 0.0
    _seeded: bool = False
    _initialized: bool = False

    @property
    def lib(self) -> ctypes.CDLL:
        """The loaded kernel library; loaded on first use so that merely constructing
        this assimilator (e.g. in the registry tests) does not require a built DART."""
        if self._lib is None:
            lib_path = self.lib_path
            if not os.path.exists(lib_path):
                raise FileNotFoundError(
                    f"DART kernel library not found: {lib_path}. Build it with "
                    "NEDAS/assim_tools/assimilators/DART/build_dart_kernels.sh, or set "
                    "assimilator_def.dart_lib to its location.")
            try:
                self._lib = load_dart_kernels(lib_path)
            except OSError as err:
                # the library built fine but something it links against is not visible here.
                # DART pulls in netCDF, and on a module-based HPC stack HDF5/MPI/CUDA underneath
                # it; those paths are usually absent once a conda environment is activated.
                # rpath in the build cannot always fix this: when an intermediate library carries
                # RUNPATH (as module-built MPI does), the loader ignores our rpath for *its*
                # dependencies, so they have to come from LD_LIBRARY_PATH at runtime.
                raise OSError(
                    f"{err}\n\nThe DART kernel library at {lib_path} could not be loaded. "
                    f"Run 'ldd {lib_path}' in this same environment to see which libraries are "
                    "missing, then add their directories to LD_LIBRARY_PATH (or load the modules "
                    "that were used to build DART).") from err
            except AttributeError as err:
                # a symbol the wrapper declares is absent: almost always a stale library left
                # from an older dart_kernels.f90
                raise AttributeError(
                    f"{err}\n\nThe DART kernel library at {lib_path} is missing an entry point "
                    "this version of NEDAS expects; rebuild it with build_dart_kernels.sh.") from err
        return self._lib

    @property
    def lib_path(self) -> str:
        return self.dart_lib or default_lib_path()

    @property
    def filter_kind_code(self) -> int:
        name = str(self.filter_kind).upper()
        try:
            return FILTER_KINDS[name]
        except KeyError:
            raise ValueError(f"unknown assimilator_def.filter_kind '{self.filter_kind}', "
                             f"choose one of {', '.join(FILTER_KINDS)}") from None

    def assimilation_algorithm(self, c):
        # Seed before any kernel runs. The seed must be the same on every rank -- the serial
        # loop has all ranks compute the increment for one global state, so a rank-dependent
        # stream would give the same observation different perturbations in different parts of
        # the domain -- and must change each cycle, or every cycle replays the same draws.
        # t2h(c.time) satisfies both: identical across ranks, advancing with the analysis time.
        if str(self.filter_kind).upper() in STOCHASTIC_KINDS:
            self.set_random_seed(self.random_seed or seed_from_time(c.time))
        super().assimilation_algorithm(c)

    def set_random_seed(self, seed: int) -> None:
        """Seed DART's random sequence (the one the stochastic kernels draw from)."""
        self.lib.dart_set_random_seed(int(seed))
        self._seeded = True

    def _ensure_seeded(self) -> None:
        """
        Never let a stochastic kernel fall back to DART's own seeding.

        That fallback builds the seed from my_task_id(), which initializes DART's utilities
        and reads input.nml -- stopping the run when the file is absent. Seeding here keeps
        the kernel usable without any DART runtime setup.
        """
        if not self._seeded and str(self.filter_kind).upper() in STOCHASTIC_KINDS:
            self.set_random_seed(self.random_seed or 1)

    def _check_gated_options(self) -> None:
        """
        sampling_error_correction changes the regression coefficient in update_from_obs_inc,
        so it would affect every filter_kind. It is refused rather than quietly ignored:
        besides the namelist flag it needs DART's correction table (a netCDF file read by
        read_sampling_error_correction) staged in the working directory, and the module
        arrays that hold it are only allocated on that path. Enabling it without that in
        place would regress with an unpopulated table.
        """
        if self.sampling_error_correction:
            raise NotImplementedError(
                "assimilator_def.sampling_error_correction is not supported yet: DART also "
                "needs its sampling error correction table (sampling_error_correction_table.nc) "
                "available at runtime, which NEDAS does not stage.")

    def _input_nml_text(self) -> str:
        """The namelist DART reads: utilities and obs_kind for init, then our kernel options."""
        return '\n'.join([
            NEDAS_NML_MARKER,
            '&utilities_nml',
            '/',
            '',
            '&assim_tools_nml',
            f'   sort_obs_inc = {_fortran_bool(self.sort_obs_inc)}',
            f'   rectangular_quadrature = {_fortran_bool(self.rectangular_quadrature)}',
            f'   gaussian_likelihood_tails = {_fortran_bool(self.gaussian_likelihood_tails)}',
            '   sampling_error_correction = .false.',
            '/',
            '',
            # required: assim_tools_init reaches obs_kind_mod via get_num_types_of_obs()
            '&obs_kind_nml',
            '/',
            '',
            '&kde_nml',
            f'   quadrature_order = {int(self.quadrature_order)}',
            '/',
            '',
        ])

    def _check_namelist_sections(self, path: str = 'input.nml') -> None:
        """
        Fail in python if a supplied namelist is missing a section DART requires.

        Without this the omission surfaces as DART calling error_handler and stopping the
        run, which leaves no exception and no traceback behind.
        """
        with open(path) as f:
            text = f.read()
        missing = [s for s in REQUIRED_NML_SECTIONS
                   if not re.search(r'&\s*' + s + r'\b', text)]
        if missing:
            raise ValueError(
                f"{os.path.abspath(path)} is missing the namelist section(s) "
                f"{', '.join('&' + s for s in missing)}, which DART requires when "
                f"filter_kind '{self.filter_kind}' initializes it. Add them (an empty "
                "section is enough), or set assimilator_def.write_input_nml to let NEDAS "
                "write the file.")

    def _probe_initialize(self) -> None:
        """
        Rehearse dart_initialize() in a throwaway interpreter.

        DART answers a bad namelist by stopping the process, which would take the whole
        NEDAS run with it. Running it in a subprocess first turns that into an ordinary
        python exception carrying DART's own message. A subprocess rather than fork(),
        because NEDAS runs under MPI and forking a rank is not safe.
        """
        code = ("import ctypes, sys\n"
                "lib = ctypes.CDLL(sys.argv[1])\n"
                "lib.dart_initialize.restype = None\n"
                "lib.dart_initialize.argtypes = []\n"
                "lib.dart_initialize()\n"
                "print('INIT-OK')\n")
        try:
            run = subprocess.run([sys.executable, '-c', code, self.lib_path],
                                 capture_output=True, text=True, cwd=os.getcwd(), timeout=120)
        except (OSError, subprocess.SubprocessError):
            return      # cannot rehearse here; fall through and try for real
        if 'INIT-OK' in run.stdout:
            return
        raise RuntimeError(
            f"DART refused to initialize in {os.getcwd()}: "
            f"{dart_error_message(run.stdout + run.stderr)}\n"
            "DART stops the process on a fatal condition, so this was checked in a "
            "subprocess; fix the namelist (or the working directory) and rerun.")

    def _ensure_initialized(self) -> None:
        """
        Put an input.nml in place and bring DART up, for the kernels that need it.

        DART reads "input.nml" from the current working directory (the name is hardcoded),
        and the sections in REQUIRED_NML_SECTIONS are not optional there. A missing file or
        section makes DART stop the process rather than return an error, which is why all of
        this happens before the kernel is called.

        An input.nml we did not write is never overwritten: it may be a real DART namelist
        whose settings matter. In that case either let NEDAS manage the file (remove it) or
        set write_input_nml to False to use yours as-is.
        """
        if self._initialized or str(self.filter_kind).upper() not in KINDS_NEEDING_INIT:
            return
        self._check_gated_options()

        if os.path.exists('input.nml'):
            with open('input.nml') as f:
                ours = NEDAS_NML_MARKER in f.readline()
            if ours and self.write_input_nml:
                with open('input.nml', 'w') as f:      # refresh, config may have changed
                    f.write(self._input_nml_text())
            elif not ours and self.write_input_nml:
                raise RuntimeError(
                    f"an input.nml not written by NEDAS is already in {os.getcwd()}; refusing "
                    "to overwrite it. Remove it to let NEDAS manage the DART namelist, or set "
                    "assimilator_def.write_input_nml to False to use it as-is.")
            else:
                self._check_namelist_sections()
        elif self.write_input_nml:
            try:
                # exclusive create: several ranks may reach this at once
                with open('input.nml', 'x') as f:
                    f.write(self._input_nml_text())
            except FileExistsError:
                pass
        else:
            raise FileNotFoundError(
                f"filter_kind '{self.filter_kind}' needs DART's namelists, read from an "
                f"input.nml in the working directory ({os.getcwd()}). Put one there providing "
                f"{', '.join('&' + s for s in REQUIRED_NML_SECTIONS)}, or set "
                "assimilator_def.write_input_nml to let NEDAS write it.")

        self._probe_initialize()
        self.lib.dart_initialize()
        self._initialized = True

    def obs_increment(self, obs_prior, obs_prior_static, obs, obs_err):
        self._ensure_initialized()
        self._ensure_seeded()
        obs_prior = np.ascontiguousarray(obs_prior, dtype=np.float64)
        obs_incr = np.empty_like(obs_prior)
        net_a = np.zeros(1)

        status = self.lib.dart_obs_increment(self.filter_kind_code, obs_prior.size, obs_prior,
                                             float(obs), float(obs_err)**2,
                                             int(bool(self.bounded_below)), int(bool(self.bounded_above)),
                                             float(self.lower_bound), float(self.upper_bound),
                                             obs_incr, net_a)
        if status:
            raise ValueError(f"DART {self.filter_kind}: {_STATUS.get(status, f'status {status}')}")

        # net_a carries over to the regression below, as it does inside DART's own
        # filter_assim; the serial loop always pairs one obs_increment with the updates
        self._net_a = float(net_a[0])
        return obs_incr

    def update_local_state(self, state_prior, state_static, obs_prior, obs_prior_static, obs_incr,
                           state_h_dist, state_v_dist, state_t_dist,
                           hroi, vroi, troi,
                           h_local_func, v_local_func, t_local_func,
                           impact_on_variable) -> None:
        # localization stays NEDAS's; lfactor[n, l] = h[l] * v[n, l] * t[n] * impact[n]
        lfactor = (h_local_func(state_h_dist, hroi)[None, :]
                   * v_local_func(state_v_dist, vroi)
                   * (t_local_func(state_t_dist, troi) * impact_on_variable)[:, None])
        self._regress(state_prior, obs_prior, obs_incr, lfactor)

    def update_local_obs(self, obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                         h_dist, v_dist, t_dist,
                         hroi, vroi, troi,
                         h_local_func, v_local_func, t_local_func,
                         impact_on_variable) -> None:
        lfactor = (h_local_func(h_dist, hroi) * v_local_func(v_dist, vroi)
                   * t_local_func(t_dist, troi) * impact_on_variable)
        # already-assimilated obs are excluded by zeroing their localization factor,
        # which is the same test the fortran loop already applies
        lfactor = np.where(used, 0.0, lfactor)
        self._regress(obs_data, obs_prior, obs_incr, lfactor)

    def _regress(self, ens, obs_prior, obs_incr, lfactor) -> None:
        """
        Regress one obs increment onto ens (nens, ...), updating it in place.

        ponytail: passes every element and lets fortran skip the ones with
        lfactor<=0, instead of subsetting to the roi first as the native EAKF does.
        The scan is cheap next to the regression itself; subset here if profiling
        ever says otherwise.
        """
        if not ens.flags['C_CONTIGUOUS'] or ens.dtype != np.float64:
            raise ValueError("DART kernels need a C-contiguous float64 ensemble to update in place")
        nens = ens.shape[0]
        flat = ens.reshape(nens, -1)          # a view, since ens is contiguous
        lfactor = np.ascontiguousarray(lfactor, dtype=np.float64).reshape(-1)

        self.lib.dart_update_from_obs_inc(nens, flat.shape[1],
                                          np.ascontiguousarray(obs_prior, dtype=np.float64),
                                          np.ascontiguousarray(obs_incr, dtype=np.float64),
                                          self._net_a, flat, lfactor)


def seed_from_time(time) -> int:
    """
    A seed derived from the analysis time: identical on every rank, different each cycle.

    Kept a positive 32-bit integer, since it is passed to fortran as a default integer.
    """
    return int(abs(int(t2h(time))) % (2**31 - 2)) + 1
