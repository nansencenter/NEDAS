import copy
import numpy as np
from NEDAS.assim_tools.assimilators.batch import BatchAssimilator

# PDAF filtertype codes (PDAF_init), restricted to the domain-localized filters:
# those are the ones whose analysis loop matches NEDAS's per-gridpoint batch loop.
# The global filters (ETKF=4, ESTKF=6, ...) and LEnKF=8 (covariance localization
# through PDAFomi_set_localize_covar, not a domain loop) are deliberately absent.
FILTER_KINDS = {'LSEIK': 3, 'LETKF': 5, 'LESTKF': 7, 'LNETF': 10, 'LKNETF': 11}

# PDAFomi weight functions (PDAFomi_init_dim_obs_l_iso locweight):
# 2 (5th-order polynomial) is Gaspari-Cohn, the same taper as NEDAS's
# gaspari_cohn_func with sradius = cradius = hroi.
LOC_WEIGHTS = {'constant': 0, 'exponential': 1, 'gaspari_cohn': 2,
               'regulated_mean': 3, 'regulated_single': 4}

# NEDAS localization_def.horizontal.type -> PDAFomi locweight, for the tapers that
# exist on both sides. 'step' has no PDAFomi counterpart (locweight 0 is constant
# weight *inside* cradius, which is the same thing -- so it does map).
NEDAS_TO_PDAF_WEIGHT = {'gaspari_cohn': 2, 'exponential': 1, 'step': 0}

def import_pypdaf():
    """
    Import pyPDAF, turning the ImportError into something actionable.

    pyPDAF is not on PyPI or conda-forge (checked 2026-09-16); it is built from source
    against a PDAF release, see install_pypdaf.md next to this file.
    """
    try:
        import pyPDAF
    except ImportError as err:
        raise ImportError(
            f"{err}\n\npyPDAF is required by the PDAF assimilator but is not importable. "
            "It has no PyPI/conda package -- build it from source (meson + a PDAF release), "
            "see NEDAS/assim_tools/assimilators/PDAF/install_pypdaf.md.") from err
    return pyPDAF

class PDAFAssimilator(BatchAssimilator):
    """
    Local ensemble filters using PDAF's own compiled analysis kernels, through pyPDAF.

    NEDAS keeps the grid, the partitioning, the obs matching and the I/O; PDAF gets one
    offline analysis per NEDAS partition and does the local-domain loop and the update
    math inside it. assimilator_def.filter_kind picks the kernel (LESTKF, LETKF, ...),
    so a new PDAF filter arrives on a pyPDAF version bump with no NEDAS code to write.

    Two mappings make this fit without giving PDAF anything NEDAS does not already have:

    1. **One PDAF analysis per partition, on MPI_COMM_SELF.** NEDAS has already assigned
       every obs within hroi of a tile to that tile (BatchAssimilator.assign_obs), so a
       partition is a self-contained analysis problem: the state it owns plus the obs that
       can reach it. Each rank therefore runs its own single-process PDAF, and the obs
       that the halo duplicates across tiles never meet in one PDAF obs vector -- which
       they would, and be assimilated twice, if PDAF's own domain decomposition
       (PDAFomi_gather_obs across a filter communicator) were used instead.

    2. **The obs priors ride along in the state vector.** PDAF wants an obs operator
       mapping state -> obs space; NEDAS has no H, it has H(x) already evaluated by each
       Dataset's obs_operator. So the PDAF state vector is [state ; obs_prior] and the obs
       operator is PDAFomi_obs_op_gridpoint picking out the appended entries (id_obs_p).
       This is PDAF's standard trick for an arbitrary/non-linear H. The appended entries
       are never part of a local analysis domain (init_dim_l only ever lists state
       entries), so they are read and discarded, never updated.

    Localization stays NEDAS's radius and taper, handed to PDAFomi as cradius/sradius and
    locweight; the parts of NEDAS's localization PDAFomi has no equivalent for (vertical,
    temporal, cross-variable impact) are rejected up front in check_capabilities() rather
    than silently dropped.
    """
    filter_kind: str = 'LESTKF'
    subtype: int = 0
    forget: float = 1.0        # PDAF's forgetting factor (multiplicative inflation)
    loc_weight: str = 'auto'   # 'auto': follow localization_def.horizontal.type
    disttype: int = -1         # -1: derive from the grid (0 cartesian, 1 periodic)
    screen: int = 0            # PDAF screen verbosity

    def check_capabilities(self, c) -> None:
        super().check_capabilities(c)
        if str(self.filter_kind).upper() not in FILTER_KINDS:
            raise ValueError(f"unknown assimilator_def.filter_kind '{self.filter_kind}', "
                             f"choose one of {', '.join(FILTER_KINDS)}")

    def check_localization_support(self, c) -> None:
        """
        Refuse the localization settings PDAFomi cannot express.

        Separate from check_capabilities() because the obs records only exist once the
        scheme has built c.obs, which is after the Context (and this assimilator) is set up.
        """
        unsupported = []
        for rec in c.obs.info.records.values():
            if np.isfinite(rec.vroi):
                unsupported.append('obs vroi (vertical localization)')
            if np.isfinite(rec.troi):
                unsupported.append('obs troi (temporal localization)')
            if any(f != 1.0 for f in rec.impact_on_variable):
                unsupported.append('obs impact_on_variable')
        if c.state.info.scalars:
            # a scalar parameter has no coordinate, so it cannot be a PDAFomi local domain
            unsupported.append('scalar state variables')
        if unsupported:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support: {', '.join(sorted(set(unsupported)))}. "
                "PDAFomi localizes by horizontal distance only; use ETKF for these.")

    def locweight_code(self, c) -> int:
        if self.loc_weight != 'auto':
            try:
                return LOC_WEIGHTS[str(self.loc_weight)]
            except KeyError:
                raise ValueError(f"unknown assimilator_def.loc_weight '{self.loc_weight}', "
                                 f"choose one of {', '.join(LOC_WEIGHTS)} or 'auto'") from None
        # 'auto': the taper NEDAS is configured with, so the PDAF analysis sees the same
        # weights the native ETKF/EAKF would have applied
        htype = str(c.config.localization_def['horizontal']['type']).lower()
        try:
            return NEDAS_TO_PDAF_WEIGHT[htype]
        except KeyError:
            raise NotImplementedError(
                f"localization_def.horizontal.type '{htype}' has no PDAFomi equivalent; "
                f"set assimilator_def.loc_weight explicitly to one of {', '.join(LOC_WEIGHTS)}.") from None

    def disttype_code(self, c) -> int:
        """
        PDAFomi distance metric. 0 = cartesian, 1 = cartesian with periodicity.

        Geographic distances (PDAFomi 2/3, coordinates in radians) are not derived
        automatically: NEDAS's Grid carries a projection, and mapping that onto PDAFomi's
        two geographic options is a guess. Set assimilator_def.disttype for those.
        """
        if self.disttype >= 0:
            return int(self.disttype)
        if getattr(c.grid, 'distance_type', 'cartesian') != 'cartesian':
            raise NotImplementedError(
                f"grid.distance_type '{c.grid.distance_type}' does not map onto a PDAFomi "
                "disttype automatically; set assimilator_def.disttype (2 or 3 for geographic, "
                "with coordinates in radians).")
        return 1 if getattr(c.grid, 'cyclic_dim', None) else 0

    def domainsize(self, c) -> np.ndarray:
        """Periodicity lengths for disttype 1; a negative entry means not periodic."""
        cyclic = str(getattr(c.grid, 'cyclic_dim', None) or '')
        return np.array([c.grid.Lx if 'x' in cyclic else -1.0,
                         c.grid.Ly if 'y' in cyclic else -1.0])

    def assimilation_algorithm(self, c) -> None:
        self.pyPDAF = import_pypdaf()
        self.check_localization_support(c)
        self._locweight = self.locweight_code(c)
        self._disttype = self.disttype_code(c)
        self._domainsize = self.domainsize(c)

        # every rank runs its own single-process PDAF over the partitions it owns, see the
        # class docstring. COMM_SELF for all four communicators, one model task, filter PE.
        from mpi4py import MPI
        comm = MPI.COMM_SELF.py2f()
        self.pyPDAF.set_parallel(comm, comm, comm, comm, 1, 1, True, 0)

        c.message = 'preparing...'
        c.state.state_post = copy.deepcopy(c.state.state_prior)

        par_list = c.state.par_list[c.pid_mem]
        c.total_tasks = len(par_list)
        c.current_task = 0
        for par_id in par_list:
            state_data = c.state.pack_local_state_data(c, par_id, c.state.state_prior,
                                                       c.state.state_z, c.state.state_static)
            obs_data = c.obs.pack_local_obs_data(c, par_id, c.obs.lobs, c.obs.lobs_prior,
                                                 c.obs.lobs_prior_static)
            nloc = state_data['state_prior'].shape[-1]
            nlobs = obs_data['x'].size
            if nloc > 0 and nlobs > 0:
                self.analyze_partition(c, state_data, obs_data)
                c.state.unpack_local_state_data(c, par_id, c.state.state_post, state_data)
            else:
                c.debug_message = f"skipped partition {par_id:7} ({nloc} state, {nlobs} obs)"
            c.current_task += 1
            c.message = f"completed {c.current_task}/{c.total_tasks} partitions."

    def local_analysis(self, c, loc_id, ind, hlfactor, state_data, obs_data):
        """
        Not used: PDAF runs the loop over local analysis domains itself, inside one
        assim_offline call per partition (see analyze_partition), so assimilation_algorithm
        never reaches BatchAssimilator's per-gridpoint loop.
        """
        raise NotImplementedError("PDAFAssimilator analyses a whole partition at a time")

    def analyze_partition(self, c, state_data: dict, obs_data: dict) -> None:
        """
        Run one PDAF offline analysis over a NEDAS partition, in place in state_data.

        The PDAF state vector is [state ; obs_prior]:
        entry n*nloc+l is field record n at location l, and the obs priors follow.
        Each local analysis domain is one location l, holding its nfld field entries.
        """
        pyPDAF = self.pyPDAF
        state_prior = state_data['state_prior']
        nens, nfld, nloc = state_prior.shape
        nlobs = obs_data['x'].size
        dim_state = nfld * nloc
        dim_p = dim_state + nlobs

        ens_prior = np.empty((dim_p, nens))
        ens_prior[:dim_state] = state_prior.reshape(nens, dim_state).T
        ens_prior[dim_state:] = obs_data['obs_prior'].T

        # obs are grouped by obs record: each record has its own hroi, and PDAFomi carries
        # cradius per obs type, not per obs
        obs_rec_ids = np.unique(obs_data['obs_rec_id'])
        obs_types = [(int(r), np.where(obs_data['obs_rec_id'] == r)[0]) for r in obs_rec_ids]

        analysis = {}   # filled by the prepoststep callback below

        def init_ens_pdaf(_filtertype, _dim_p, _dim_ens, state_p, uinv, ens_p, status):
            ens_p[:] = ens_prior
            return state_p, uinv, ens_p, status

        def init_n_domains_pdaf(_step, _ndomains):
            return nloc

        def init_dim_l_pdaf(_step, domain_p, _dim_l):
            # domain_p is 1-based, and so are the state indices PDAFlocal expects
            loc_id = domain_p - 1
            ids = (np.arange(nfld) * nloc + loc_id + 1).astype(np.intc)
            pyPDAF.PDAFlocal.set_indices(nfld, ids)
            return nfld

        def init_dim_obs_pdafomi(_step, _dim_obs):
            dim_obs = 0
            for i_obs, (obs_rec_id, ind) in enumerate(obs_types, start=1):
                pyPDAF.PDAFomi.set_doassim(i_obs, 1)
                pyPDAF.PDAFomi.set_disttype(i_obs, self._disttype)
                pyPDAF.PDAFomi.set_ncoord(i_obs, 2)
                # identity obs operator onto the appended obs_prior entries (1-based)
                id_obs_p = np.zeros((1, ind.size), dtype=np.intc, order='F')
                id_obs_p[0] = dim_state + ind + 1
                pyPDAF.PDAFomi.set_id_obs_p(i_obs, 1, ind.size, id_obs_p)
                pyPDAF.PDAFomi.set_use_global_obs(i_obs, 1)
                if self._disttype == 1:
                    pyPDAF.PDAFomi.set_domainsize(i_obs, 2, self._domainsize)

                ocoord_p = np.zeros((2, ind.size), order='F')
                ocoord_p[0] = obs_data['x'][ind]
                ocoord_p[1] = obs_data['y'][ind]
                # PDAF works with the inverse obs error variance (diagonal R)
                ivar_obs_p = 1.0 / obs_data['err_std'][ind]**2
                dim_obs += pyPDAF.PDAFomi.gather_obs(i_obs, ind.size,
                                                     obs_data['obs'][ind], ivar_obs_p,
                                                     ocoord_p, 2, self.hroi(obs_data, obs_rec_id))
            return dim_obs

        def obs_op_pdafomi(_step, _dim_p, _dim_obs_p, state_p, ostate):
            for i_obs in range(1, len(obs_types) + 1):
                ostate = pyPDAF.PDAFomi.obs_op_gridpoint(i_obs, state_p, ostate)
            return ostate

        def init_dim_obs_l_pdafomi(domain_p, _step, _dim_obs, _dim_obs_l):
            loc_id = domain_p - 1
            coords_l = np.array([state_data['x'][loc_id], state_data['y'][loc_id]])
            dim_obs_l = 0
            for i_obs, (obs_rec_id, _) in enumerate(obs_types, start=1):
                hroi = self.hroi(obs_data, obs_rec_id)
                # cradius = sradius = hroi: PDAFomi tapers to zero at sradius, which is
                # where NEDAS's localization functions taper to zero too
                dim_obs_l += pyPDAF.PDAFomi.init_dim_obs_l_iso(i_obs, coords_l, self._locweight,
                                                               hroi, hroi, dim_obs_l)
            return dim_obs_l

        def prepoststep_pdaf(step, _dim_p, _dim_ens, _dim_ens_p, _dim_obs_p,
                             state_p, uinv, ens_p, _flag):
            # called twice by assim_offline, before (step<0) and after the analysis
            if step >= 0:
                analysis['ens'] = ens_p.copy()
            return state_p, uinv, ens_p

        param_int = np.array([dim_p, nens], dtype=np.intc)
        param_real = np.array([float(self.forget)])
        _, _, status = pyPDAF.init(FILTER_KINDS[str(self.filter_kind).upper()],
                                   int(self.subtype), 0,
                                   param_int, param_int.size, param_real, param_real.size,
                                   init_ens_pdaf, int(self.screen))
        if status != 0:
            raise RuntimeError(f"PDAF_init failed with status {status}")
        try:
            pyPDAF.PDAFomi.init(len(obs_types))
            pyPDAF.PDAFomi.init_local()
            status = pyPDAF.assim_offline(init_dim_obs_pdafomi, obs_op_pdafomi,
                                          init_n_domains_pdaf, init_dim_l_pdaf,
                                          init_dim_obs_l_pdafomi, prepoststep_pdaf, 0)
            if status != 0:
                raise RuntimeError(f"PDAF analysis failed with status {status}")
        finally:
            # PDAF holds the ensemble and obs arrays for this partition's dimensions;
            # the next partition has different ones, so it starts from a clean PDAF
            pyPDAF.PDAF.deallocate()

        if 'ens' not in analysis:
            raise RuntimeError("PDAF returned no analysis ensemble (prepoststep was not called)")
        state_prior[:] = analysis['ens'][:dim_state].T.reshape(nens, nfld, nloc)

    @staticmethod
    def hroi(obs_data: dict, obs_rec_id: int) -> float:
        return float(obs_data['hroi'][obs_rec_id])
