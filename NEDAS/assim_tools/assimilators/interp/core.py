import numpy as np
from NEDAS.assim_tools.assimilators.batch import BatchAssimilator
from NEDAS.assim_tools.localization.distance_based import gaspari_cohn_func

METHODS = ('cressman', 'optimal')


def cressman_weight(dist, roi):
    """Cressman (1959) objective-analysis weight: W(r) = (R^2 - r^2) / (R^2 + r^2) for r < R,
    else 0. Purely distance-based -- NO observation-error term (each obs weighted independently
    of every other obs, unlike optimal interpolation -- see oi_value below)."""
    dist = np.asarray(dist, dtype=float)
    w = np.zeros(dist.shape)
    valid = dist < roi
    w[valid] = (roi**2 - dist[valid]**2) / (roi**2 + dist[valid]**2)
    return w


def max_angular_gap(target_x, target_y, obs_x, obs_y):
    """Largest empty azimuthal wedge (radians) around a target point, given the obs currently
    considered local to it. 2*pi if there are 0 or 1 obs (no bearing spread at all); near 0 for
    obs evenly surrounding the point on all sides.

    2026-07-30, Yue: "when all available obs is to the same side, the interpolation result is
    really bad ... (it's basically extrapolation)". A local system can be well-conditioned and
    still be a bad estimate if every contributing obs sits in a narrow wedge on one side --
    neither cressman_value nor oi_value's plain distance-based weighting notices this on its
    own, so it's checked explicitly here and used as an independent reject criterion (see
    max_gap_deg in cressman_value/oi_value/interp_value) alongside roi."""
    if len(obs_x) <= 1:
        return 2 * np.pi
    theta = np.sort(np.arctan2(obs_y - target_y, obs_x - target_x) % (2 * np.pi))
    gaps = np.diff(np.concatenate([theta, theta[:1] + 2 * np.pi]))
    return np.max(gaps)


def cressman_value(target_x, target_y, obs_x, obs_y, obs_val, roi, max_gap_deg=None):
    """Cressman-weighted average at ONE target point. NaN if no obs within roi, or (when
    max_gap_deg is set) if the in-range obs are confined to too narrow an azimuthal wedge around
    the point -- see max_angular_gap's docstring."""
    dist = np.hypot(obs_x - target_x, obs_y - target_y)
    w = cressman_weight(dist, roi)
    valid = w > 0
    if max_gap_deg is not None and np.degrees(max_angular_gap(target_x, target_y, obs_x[valid], obs_y[valid])) > max_gap_deg:
        return np.nan
    wsum = np.sum(w)
    if wsum <= 0:
        return np.nan
    return np.sum(w * obs_val) / wsum


def oi_value(target_x, target_y, obs_x, obs_y, obs_val, obs_err, roi, max_gap_deg=None):
    """Optimal-Interpolation-style (Gandin) value at ONE target point, from local obs alone (no
    background/ensemble blend) -- see InterpolationAssimilator's docstring below for the full
    derivation/rationale.

    Solves (Cov + R) w = c for the local obs within roi of the target point, where
    Cov_ij = sigma_b^2 * rho(d_ij) (rho = gaspari_cohn_func, an isotropic correlation function),
    R = diag(obs_err_i^2), c_i = sigma_b^2 * rho(d_i), sigma_b^2 = local obs sample variance
    (self-calibrating, no extra tunable parameter). Returns interp_val = w . obs_val.

    As obs_err -> 0 this reduces to pure Kriging (exact recovery at each obs's own location,
    2026-07-30, Yue's requested property) -- unlike cressman_value above, which can never have
    this property since it weights each obs independently rather than solving a joint system.
    Points with no obs within roi get NaN (caller's responsibility to fill, e.g. np.nan_to_num,
    same convention as NEDAS.grid.IrregularGrid); same when max_gap_deg is set and the in-range
    obs are confined to too narrow an azimuthal wedge (see max_angular_gap's docstring) -- being
    well-conditioned numerically doesn't mean a one-sided local system is a good ESTIMATE, it's
    still extrapolation past whatever's actually observed, just extrapolation the matrix solve
    happens to be able to complete.

    Obs-obs/obs-point distances are plain Euclidean (not cyclic-boundary-aware) -- fine as long
    as roi is small relative to the domain (obs pairs within one target point's local
    neighborhood essentially never need to wrap around a periodic boundary from each other).
    """
    d_op = np.hypot(obs_x - target_x, obs_y - target_y)
    valid = np.where(d_op < roi)[0]
    if len(valid) == 0:
        return np.nan
    if max_gap_deg is not None and np.degrees(max_angular_gap(target_x, target_y, obs_x[valid], obs_y[valid])) > max_gap_deg:
        return np.nan
    if len(valid) == 1:
        return obs_val[valid[0]]  # trivial case, matrix solve degenerates to identity

    ox, oy, oval, oerr = obs_x[valid], obs_y[valid], obs_val[valid], obs_err[valid]
    d_oo = np.hypot(ox[:, None] - ox[None, :], oy[:, None] - oy[None, :])
    sigma_b2 = np.var(oval)
    if sigma_b2 <= 0:
        sigma_b2 = 1.0  # degenerate (all local obs identical) -- nominal scale
    Cov = sigma_b2 * gaspari_cohn_func(d_oo, roi)
    R = np.diag(oerr**2)
    c_vec = sigma_b2 * gaspari_cohn_func(d_op[valid], roi)
    try:
        w = np.linalg.solve(Cov + R, c_vec)
    except np.linalg.LinAlgError:
        return np.nan
    return np.dot(w, oval)


def interp_value(target_x, target_y, obs_x, obs_y, obs_val, obs_err, roi, method='optimal',
                 max_gap_deg=None):
    """Dispatch to cressman_value or oi_value by name -- InterpolationAssimilator's single
    entry point, so a 'method' config option picks the same algorithm consistently.

    max_gap_deg (None by default -- backward compatible, no behavior change unless set): reject
    (return NaN) if the local obs' azimuthal coverage around the target point leaves an empty
    wedge wider than this many degrees -- see max_angular_gap's docstring. A one-sided obs
    cluster is extrapolation, not interpolation, regardless of how well-conditioned the
    underlying weighted-average/matrix-solve is."""
    if method == 'cressman':
        return cressman_value(target_x, target_y, obs_x, obs_y, obs_val, roi, max_gap_deg=max_gap_deg)
    elif method == 'optimal':
        return oi_value(target_x, target_y, obs_x, obs_y, obs_val, obs_err, roi, max_gap_deg=max_gap_deg)
    else:
        raise NotImplementedError(f"interp_value: unknown method '{method}', choose from {METHODS}")


class InterpolationAssimilator(BatchAssimilator):
    """Batch assimilator that replaces the observed state variable's prior with an
    Optimal-Interpolation-style (Gandin) local analysis of the observations alone -- no
    background/ensemble blend -- instead of computing a Kalman gain. Every ensemble member gets
    the SAME analyzed value at each grid point (no ensemble-dependent update), by design: this
    assimilator exists to build a clean, obs-only alignment TARGET for AlignmentUpdator, used by
    the 'OA' align-then-filter scheme, not to produce a spread-preserving posterior on its own --
    that's iter1's job
    (a normal ETKFAssimilator + AdditiveUpdator running on the aligned prior). Deliberately NOT
    blended with a real background/ensemble state, unlike textbook OI --
    that would reintroduce the same "aligning toward a state-dependent blend" problem that made
    the original MSA design (aligning toward the ETKF's own posterior) fail (see this
    experiment's real_target_alignment_test.py offline finding).

    2026-07-30 (Yue): "specify obs_err=0 to recover exactly the observed value at obs locations,
    the interpolation part is the unobserved locations will be based on an isotropic error
    covariance model, distance based, so the observation will receive less weight" -- a plain
    per-obs weighted average (Cressman, Barnes, or the earlier draft's GC+1/err^2 hybrid) can
    NEVER have this exact-recovery property, since each obs is weighted independently of the
    others. Getting it requires solving a linear system over the LOCAL obs jointly -- see
    oi_value above for the full derivation: as obs_err -> 0 this reduces to pure Kriging (exact
    recovery at each obs's own location); at nonzero obs_err, added "nugget" variance shrinks
    each obs's weight, and points far from any obs decay toward zero.

    method (2026-07-30, Yue: "make the interpolation schemes configurable too, method=cressman
    or optimal"): 'optimal' (default) is the OI solve described above; 'cressman' is the simpler,
    cheaper Cressman (1959) distance-weighted average (cressman_value above) -- no matrix solve,
    no exact-recovery-at-obs-locations property, each obs weighted independently. Both dispatch
    through interp_value above.

    State variables/levels with no obs impact (obs_data['impact_on_variable']==0, e.g. theta/q/
    pstar when only wind is observed) are left unchanged (pass-through) -- the experiment's
    obs_def must set impact_on_variable to 0 for every variable this assimilator should NOT
    overwrite (unlike ETKF, a naive value-substitution scheme cannot sensibly balance an observed
    variable's obs against an unrelated state variable the way a real Kalman gain can).

    max_gap_deg (2026-07-30, Yue: "when all available obs is to the same side, the interpolation
    result is really bad ... is there a way to ignore them?" -- "(it's basically extrapolation)")
    -- None (default, no behavior change) skips this check entirely. Set to a degree threshold
    (e.g. 270) to reject (leave state_prior unchanged, same as "no obs in range") any point whose
    local obs are confined to too narrow an azimuthal wedge -- see max_angular_gap above.
    """
    method: str = 'optimal'
    max_gap_deg: float | None = None

    def local_analysis(self, c, loc_id, ind, hlfactor, state_data, obs_data):
        state_var_id = state_data['var_id']
        obs_rec_id = obs_data['obs_rec_id'][ind]
        impact_on_variable = obs_data['impact_on_variable'][:, state_var_id][obs_rec_id]
        obs_value = obs_data['obs'][ind]
        obs_err = obs_data['err_std'][ind]
        roi = obs_data['hroi'][obs_rec_id]
        ox, oy = obs_data['x'][ind], obs_data['y'][ind]
        state_x, state_y = state_data['x'][loc_id], state_data['y'][loc_id]

        nfld = state_data['state_prior'].shape[1]
        for n in range(nfld):
            valid = np.where(impact_on_variable[:, n] > 0)[0]
            if len(valid) == 0:
                continue  # no obs impact this field -- leave state_prior[:, n, loc_id] unchanged

            interp_val = interp_value(state_x, state_y, ox[valid], oy[valid], obs_value[valid],
                                      obs_err[valid], float(roi[valid][0]), method=self.method,
                                      max_gap_deg=self.max_gap_deg)
            if np.isnan(interp_val):
                continue  # no obs in range / too one-sided / ill-conditioned -- leave unchanged

            # same analyzed value for every ensemble member (deterministic, no spread update)
            state_data['state_prior'][:, n, loc_id] = interp_val
