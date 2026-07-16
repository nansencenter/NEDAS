from abc import ABC, abstractmethod
import numpy as np
from .context import Context

class Preconditioner(ABC):
    """
    Base class for preconditioners: an optional step that runs before the assimilator, blending
    state and obs to condition the prior (e.g. correcting position error) before the assimilator's
    own amplitude-only correction runs.

    Unlike Transform (which only ever sees state OR obs, one field/obs-record at a time, and is
    called from within state.py/obs.py before this cycle's obs are even fully prepared), a
    Preconditioner runs later in filter_iter -- after both c.state.fields_prior and
    c.obs.obs_prior/obs_seq are populated -- so it can legitimately blend state and obs together.

    A Preconditioner is a pre-assimilation-only concept (hence the name) and has no "undo" method
    of its own; if pre_assimilate warps state, restoring the original coordinate frame afterward
    is the Scheme driver's bookkeeping, handled by the free function restore_precondition below
    (called from filter.py after the assimilator runs, before the updator).
    """
    #: Whether pre_assimilate is a no-op given the current config. Mirrors Transform.is_identity's
    #: convention/purpose.
    is_identity: bool = False

    #: per-(mem_id, rec_id) displacement fields computed by pre_assimilate, consumed by
    #: restore_precondition to undo the warp. Subclasses that don't warp state may leave this empty.
    displace: dict

    def __init__(self, c: Context, **kwargs) -> None:
        self.displace = {}

    @abstractmethod
    def pre_assimilate(self, c: Context) -> None:
        """
        Runs once per outer-loop iteration, after c.state.fields_prior and c.obs.obs_prior/
        obs_seq are prepared, but before c.assimilator.assimilate(c). May modify
        c.state.fields_prior in place; if it does, it is responsible for refreshing
        c.obs.obs_prior (via c.obs.prepare_obs_from_state(c, 'prior')) before returning, so the
        assimilator sees obs priors consistent with the modified state.
        """
        ...


def restore_precondition(c: Context) -> None:
    """
    Undo any state warp applied by c.preconditioner.pre_assimilate, using the displacement it
    recorded in c.preconditioner.displace. Called from the Scheme driver after
    c.assimilator.assimilate(c), before c.updator.update(c) -- both c.state.fields_prior and
    c.state.fields_post are in "preconditioned" space at that point, so both get un-warped here,
    leaving the updator to see physical-space fields exactly as it would with no preconditioner
    at all. A no-op if the configured preconditioner never populated self.displace (e.g. Identity).
    """
    displace = c.preconditioner.displace
    if not displace:
        return
    from NEDAS.utils.optical_flow import warp
    for (mem_id, rec_id), d in displace.items():
        if (mem_id, rec_id) in c.state.fields_prior:
            fld = c.state.fields_prior[mem_id, rec_id]
            c.state.fields_prior[mem_id, rec_id] = warp(fld, d[0], d[1])
        if (mem_id, rec_id) in c.state.fields_post:
            fld = c.state.fields_post[mem_id, rec_id]
            c.state.fields_post[mem_id, rec_id] = warp(fld, d[0], d[1])
