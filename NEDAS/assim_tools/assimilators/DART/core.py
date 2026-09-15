import ctypes
import os
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

# status codes returned by dart_obs_increment
_STATUS = {1: "obs error variance and prior spread are both zero",
           2: "unknown filter_kind",
           3: "likelihood underflowed (BNRHF); check the bounds"}

# Kernels that need DART's utilities subsystem initialized, which this interface does not
# do. Refused here rather than allowed to abort the process: DART answers an unmet
# precondition with error_handler, which terminates the run outright and cannot be caught.
#
# Only KDE remains. It reads its own kde_nml, and find_namelist_in_file refuses to run
# until initialize_utilities() has been called. Enabling it means adding an initialization
# entry point, an input.nml in every rank's working directory (the name is hardcoded and
# cwd-relative in utilities_mod), and accepting the dart_log.out/dart_log.nml DART writes
# there. Verified to work once those are in place, so this is a setup gap, not a limitation
# of the kernel.
#
# ENKF and KERNEL used to be listed here for what looked like the same reason. They are not:
# their trigger was my_task_id() inside their own seeding block, which initializes the
# utilities as a side effect. Seeding explicitly via dart_set_random_seed() skips that block
# entirely, so they need no initialization and no input.nml.
UNSUPPORTED_KINDS = {
    'KDE': "needs DART's utilities initialized before it can read kde_nml",
}

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
    return lib

def default_lib_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'libdartkernels.so')

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

    Static members (covariance_def.nens_static) are not supported: DART's kernels take the
    dynamic ensemble alone. check_capabilities() rejects that configuration up front, so the
    static arguments below are accepted to satisfy the SerialAssimilator interface and ignored.
    """
    dart_lib: str = ''
    filter_kind: str = 'EAKF'
    random_seed: int = 0          # 0: derive a seed from the analysis time
    bounded_below: bool = False
    bounded_above: bool = False
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    _lib = None
    _net_a: float = 0.0
    _seeded: bool = False

    @property
    def lib(self) -> ctypes.CDLL:
        """The loaded kernel library; loaded on first use so that merely constructing
        this assimilator (e.g. in the registry tests) does not require a built DART."""
        if self._lib is None:
            lib_path = self.dart_lib or default_lib_path()
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
        return self._lib

    @property
    def filter_kind_code(self) -> int:
        name = str(self.filter_kind).upper()
        if name in UNSUPPORTED_KINDS:
            raise NotImplementedError(
                f"assimilator_def.filter_kind '{name}' is not supported: {UNSUPPORTED_KINDS[name]}")
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

    def obs_increment(self, obs_prior, obs_prior_static, obs, obs_err):
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
