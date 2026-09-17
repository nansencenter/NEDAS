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

# PDAF can only be initialized once per process: a second PDAF_init segfaults, with or without
# a PDAF_deallocate in between (checked against PDAF V3.0 / pyPDAF 1.0.4, 2026-09-17), while
# assim_offline can be called as often as we like on that one init. So the init is process-wide
# -- shared by every partition, outer-loop iteration and analysis cycle -- and has to be sized
# for the largest state vector this rank will ever hand over; see ensure_initialized().
_pdaf = {'initialized': False, 'dim_p': 0, 'nens': 0, 'filter_kind': ''}

def import_pypdaf():
    """
    Import pyPDAF, turning the ImportError into something actionable.

    Called where it is needed rather than cached on the assimilator: in offline io_mode the
    scheduler pickles the assimilator out to worker processes, and a module attribute makes
    that fail with "cannot pickle 'module' object" (found in the first offline L96 run,
    2026-09-17). Python's own module cache makes the repeated import free.

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
       (PDAFomi_gather_obs across a filter communicator) were used instead. PDAF itself is
       initialized once per process, since it cannot be initialized twice, and every partition
       reuses that one instance -- see ensure_initialized() and _pdaf above.

    2. **The obs operator serves NEDAS's obs_prior.** PDAF wants an obs operator mapping
       state -> obs space; NEDAS has no H, it has H(x) already evaluated per member by each
       Dataset's obs_operator. So obs_op_pdafomi ignores the state it is handed and returns
       the stored obs prior, reading which member that is off a tag carried in the last entry
       of the state vector -- rather than assuming the order PDAF loops over members in. The
       tag belongs to no local analysis domain, so PDAF reads it and never writes it.

    Localization stays NEDAS's radius and taper, handed to PDAFomi as cradius/sradius and
    locweight; the parts of NEDAS's localization PDAFomi has no equivalent for (vertical,
    temporal, cross-variable impact) are rejected up front in check_localization_support()
    rather than silently dropped.

    This runs PDAF's formulation, not a version of it adjusted to agree with NEDAS's own ETKF.
    One difference is known and left standing: PDAF applies the taper as textbook
    R-localization (Hunt et al. 2007), scaling the inverse obs error variance by the weight w,
    whereas NEDAS's ETKF multiplies the whitened obs anomalies and the innovation by w
    (ensemble_transform_weights), so its taper enters the analysis as w^2 -- a tighter
    effective localization from the same hroi, and not the convention NEDAS's own EAKF uses
    either. That is a result this interface exists to produce: what a given code's own choices
    do to a method the literature calls the same. Everything else is identical -- with
    localization off the two agree to 1e-15, and with the taper held fixed to 1e-16, which is
    what makes tests/test_pdaf_letkf.py a regression test on upstream's numerics.
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
        return 1 if any(self.cyclic_axes(c)) else 0

    @staticmethod
    def cyclic_axes(c) -> tuple:
        """
        Which of x, y wrap around, for both of NEDAS's grid classes.

        Grid (2D) carries cyclic_dim, a string ('x', 'y', 'xy' or None); Grid1D carries a plain
        cyclic flag and has no y. Reading only cyclic_dim silently localized Lorenz-96 -- a ring
        -- as if it had two open ends, which is how this was found (2026-09-17, first end-to-end
        L96 run): NEDAS's own distance wrapped, PDAF's did not, so the two disagreed only near
        x=0 and x=Lx.
        """
        grid = c.grid
        if hasattr(grid, 'cyclic_dim'):
            cyclic = str(grid.cyclic_dim or '')
            return 'x' in cyclic, 'y' in cyclic
        return bool(getattr(grid, 'cyclic', False)), False

    def domainsize(self, c) -> np.ndarray:
        """Periodicity lengths for disttype 1; a negative entry means not periodic."""
        cyclic_x, cyclic_y = self.cyclic_axes(c)
        return np.array([c.grid.Lx if cyclic_x else -1.0,
                         c.grid.Ly if cyclic_y else -1.0])

    def assimilation_algorithm(self, c) -> None:
        import_pypdaf()      # fail early, and with a useful message, if it is not installed
        self.check_localization_support(c)
        self._locweight = self.locweight_code(c)
        self._disttype = self.disttype_code(c)
        self._domainsize = self.domainsize(c)

        c.message = 'preparing...'
        c.state.state_post = copy.deepcopy(c.state.state_prior)

        par_list = c.state.par_list[c.pid_mem]
        nloc_max = self.max_locations(c)
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
                nfld = state_data['state_prior'].shape[1]
                self.ensure_initialized(c, nfld * nloc_max)
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

    def max_locations(self, c) -> int:
        """Unmasked grid points in the largest partition this rank owns (see ensure_initialized)."""
        counts = []
        for par_id in c.state.par_list[c.pid_mem]:
            if len(c.grid.x.shape) == 2:
                ist, ied, di, jst, jed, dj = c.state.partitions[par_id]
                msk = c.grid.mask[jst:jed:dj, ist:ied:di]
            else:
                msk = c.grid.mask[c.state.partitions[par_id]]
            counts.append(int(np.sum(~msk)))
        return max(counts, default=0)

    def ensure_initialized(self, c, dim_state_max: int) -> None:
        """
        Bring PDAF up, once for the whole process (see the _pdaf comment above).

        The ensemble PDAF allocates here is never the one we analyse: each partition injects
        its own in the prestep callback. What matters is that dim_p is large enough for every
        partition this rank will hand over, and that dim_ens and the filter never change.
        """
        dim_p = dim_state_max + 1      # + the member tag, see analyze_partition
        filter_kind = str(self.filter_kind).upper()
        if _pdaf['initialized']:
            if dim_p > _pdaf['dim_p'] or c.nens != _pdaf['nens'] or filter_kind != _pdaf['filter_kind']:
                raise RuntimeError(
                    f"PDAF is already initialized in this process for filter {_pdaf['filter_kind']}, "
                    f"dim_p {_pdaf['dim_p']}, {_pdaf['nens']} members, and cannot be initialized "
                    f"again (a second PDAF_init crashes); this analysis needs filter {filter_kind}, "
                    f"dim_p {dim_p}, {c.nens} members. Changing the ensemble size, the partitioning "
                    "or assimilator_def.filter_kind mid-run is therefore not supported.")
            return

        pyPDAF = import_pypdaf()
        # PDAF runs entirely inside this rank: NEDAS has already made each partition a
        # self-contained analysis problem, so all four communicators are MPI_COMM_SELF and
        # PDAF sees one filter PE with the whole (partition-sized) state.
        from mpi4py import MPI
        comm = MPI.COMM_SELF.py2f()
        pyPDAF.set_parallel(comm, comm, comm, comm, 1, 1, True, 0)

        def init_ens_pdaf(_filtertype, _dim_p, _dim_ens, state_p, uinv, ens_p, status):
            ens_p[:] = 0.0
            return state_p, uinv, ens_p, status

        param_int = np.array([dim_p, c.nens], dtype=np.intc)
        param_real = np.array([float(self.forget)])
        _, _, status = pyPDAF.init(FILTER_KINDS[filter_kind], int(self.subtype), 0,
                                   param_int, param_int.size, param_real, param_real.size,
                                   init_ens_pdaf, int(self.screen))
        if status != 0:
            raise RuntimeError(f"PDAF_init failed with status {status}")
        _pdaf.update(initialized=True, dim_p=dim_p, nens=c.nens, filter_kind=filter_kind)

    def analyze_partition(self, c, state_data: dict, obs_data: dict) -> None:
        """
        Run one PDAF analysis over a NEDAS partition, in place in state_data.

        The PDAF state vector is the partition's state, entry n*nloc+l being field record n at
        location l, padded out to the process-wide dim_p, with the member index in the last
        entry (the tag, see obs_op_pdafomi). Each local analysis domain is one location l,
        holding its nfld field entries; the padding and the tag belong to no domain, so PDAF
        reads them and never writes them.
        """
        pyPDAF = import_pypdaf()
        state_prior = state_data['state_prior']
        nens, nfld, nloc = state_prior.shape
        dim_state = nfld * nloc
        dim_p = _pdaf['dim_p']
        tag = dim_p - 1

        # obs are grouped by obs record: each record has its own hroi, and PDAFomi carries
        # cradius per obs type, not per obs
        obs_rec_ids = np.unique(obs_data['obs_rec_id'])
        obs_types = [(int(r), np.where(obs_data['obs_rec_id'] == r)[0]) for r in obs_rec_ids]
        obs_prior = np.ascontiguousarray(obs_data['obs_prior'], dtype=np.float64)

        analysis = {'called': False}   # filled by the prepoststep callback below

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
                # id_obs_p is what PDAFomi's own obs operators use to pick observed entries out
                # of the state vector; ours does not go through them (see obs_op_pdafomi), so
                # these are dummies -- but OMI wants the array set, so they have to be valid.
                id_obs_p = np.ones((1, ind.size), dtype=np.intc, order='F')
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
            """
            H(x) for one ensemble member -- read out of NEDAS's obs_prior rather than computed.

            NEDAS has no H to give PDAF: each Dataset evaluates its own obs operator per member,
            long before the analysis. So this hands PDAF the obs prior NEDAS already has, and
            the member it belongs to is read off the tag in the state vector rather than assumed
            from the order PDAF happens to loop in.
            """
            member = float(state_p[tag])
            m = int(round(member))
            if abs(member - m) > 1e-9 or not 0 <= m < nens:
                raise RuntimeError(
                    f"PDAF asked for H(x) of a state vector whose member tag is {member}: it is "
                    "applying the obs operator to something other than a single ensemble member "
                    "(the ensemble mean, say), which this interface cannot serve from obs_prior.")
            for i_obs, (_, ind) in enumerate(obs_types, start=1):
                ostate = pyPDAF.PDAFomi.gather_obsstate(
                    i_obs, np.ascontiguousarray(obs_prior[m, ind]), ostate)
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

        def prepoststep_pdaf(_step, _dim_p, _dim_ens, _dim_ens_p, _dim_obs_p,
                             state_p, uinv, ens_p, _flag):
            # assim_offline calls this exactly twice, once before and once after the analysis.
            # Which is which is taken from the call order, not from the sign of step: offline,
            # PDAF passes step 0 to the pre-analysis call, not the negative step the online
            # interface documents.
            #
            # The pre-analysis call is where this partition's ensemble goes in: PDAF holds one
            # ensemble array for the whole process (it can only be initialized once), so every
            # partition writes its own state into it here -- ens_p is a view on PDAF's own
            # array, so the write lands in Fortran -- and reads the analysis back out after.
            if not analysis['called']:
                ens_p[:] = 0.0
                ens_p[:dim_state] = state_prior.reshape(nens, dim_state).T
                ens_p[tag] = np.arange(nens)
                analysis['called'] = True
            else:
                analysis['ens'] = ens_p.copy()
            return state_p, uinv, ens_p

        pyPDAF.PDAFomi.init(len(obs_types))
        pyPDAF.PDAFomi.init_local()
        status = pyPDAF.assim_offline(init_dim_obs_pdafomi, obs_op_pdafomi,
                                      init_n_domains_pdaf, init_dim_l_pdaf,
                                      init_dim_obs_l_pdafomi, prepoststep_pdaf, 0)
        if status != 0:
            raise RuntimeError(f"PDAF analysis failed with status {status}")
        if 'ens' not in analysis:
            raise RuntimeError("PDAF returned no analysis ensemble: prepoststep ran "
                               f"{1 if analysis['called'] else 0} time(s), not twice")
        state_prior[:] = analysis['ens'][:dim_state].T.reshape(nens, nfld, nloc)

    @staticmethod
    def hroi(obs_data: dict, obs_rec_id: int) -> float:
        return float(obs_data['hroi'][obs_rec_id])
