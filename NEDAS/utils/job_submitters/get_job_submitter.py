from typing import Optional
from .base import JobSubmitter
from .slurm import SLURMJobSubmitter
from .oar import OARJobSubmitter

def get_job_submitter(scheduler:Optional[str]=None, **kwargs) -> JobSubmitter:
    """
    Return the correct JobSubmitter class instance given the scheduler type
    """
    submitters = {
        'slurm': SLURMJobSubmitter,
        'oar':   OARJobSubmitter,
        }

    if scheduler is None:
        ##use the vanila JobSubmitter if scheduler is not specified
        return JobSubmitter(**kwargs)

    else:
        submitter = submitters.get(scheduler.lower())
        if not submitter:
            raise ValueError(f"Unsupported scheduler type: {scheduler}")
        return submitter(**kwargs)

