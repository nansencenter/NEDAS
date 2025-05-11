import importlib
from typing import Optional
from NEDAS.job_submitters.base import JobSubmitter

registry_by_host = {
    'macos': 'MacOSJobSubmitter',
    'betzy': 'BetzyJobSubmitter',
}
registry_by_scheduler = {
    'slurm': 'SLURMJobSubmitter',
    'oar': 'OARJobSubmitter',
}

def get_job_submitter(host: Optional[str]=None,
                      scheduler:Optional[str]=None,
                      **kwargs) -> JobSubmitter:
    """
    Factory function to get the JobSubmitter instance.

    Based on the input args `host` or `scheduler`, if an implemented JobSubmitter
    subclass can be located, return an instance of that subclass.
    If not, will use the base JobSubmitter class instance.
    
    Args:
        host (str, optional): Host machine name.

    Returns:
        JobSubmitter: An instance of the corresponding JobSubmitter subclass.
    """
    if host:
        host_name = host.lower()
        if host_name in registry_by_host.keys():
            module = importlib.import_module('NEDAS.job_submitters.'+host_name)
            JobSubmitterClass = getattr(module, registry_by_host[host_name])
            return JobSubmitterClass(**kwargs)

    if scheduler:
        scheduler_name = scheduler.lower()
        if scheduler_name in registry_by_scheduler.keys():
            module = importlib.import_module('NEDAS.job_submitters.'+scheduler_name)
            JobSubmitterClass = getattr(module, registry_by_scheduler[scheduler_name])
            return JobSubmitterClass(**kwargs)

    return JobSubmitter(**kwargs)
