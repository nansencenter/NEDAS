import os
import inspect
from abc import ABC, abstractmethod
import numpy as np
from NEDAS.config import parse_config
from .context import Context
from .types import MemID, FieldRecordID

class Updator(ABC):
    """
    Base class for updators of the model restart files
    """
    increment: dict = {}

    def __init__(self, c: Context):
        ##get updator parameters from config file
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, parse_args=False, **c.config.updator_def)
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update(self, c: Context) -> None:
        """
        Top-level routine to apply the analysis increments to the original model
        restart files (as initial conditions for the next forecast)
        """
        mem_list = c.mem_list
        rec_list = c.state.rec_list
        pid_mem_show = [p for p,lst in mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in rec_list.items() if len(lst)>0][0]
        c.pid_show = pid_rec_show * c.config.nproc_mem + pid_mem_show

        c.print_1p(f'>>> update model restart files with analysis increments\n')

        ##compute analysis increments
        self.compute_increment(c)

        ##process the fields, each processor goes through its own subset of
        ##mem_id,rec_id simultaneously
        ##but need to keep every rank in sync to coordinate multiprocess file access
        nm_max = np.max([len(lst) for _,lst in mem_list.items()])
        nr_max = np.max([len(lst) for _,lst in rec_list.items()])
        for r in range(nr_max):
            for m in range(nm_max):
                pid_active = ( m < len(mem_list[c.pid_mem]) and r < len(rec_list[c.pid_rec]) )

                #TODO: there is a bug here -> UCX WARN raised.
                #self.init_file_locks(c, r, m, pid_active)

                if pid_active:
                    mem_id = mem_list[c.pid_mem][m]
                    rec_id = rec_list[c.pid_rec][r]
                    rec = c.state.info.fields[rec_id].asdict()
                    c.show_progress(f"PID {c.pid:4}: update_restartfile mem{mem_id+1:03} '{rec['name']:20}' {rec['time']} k={rec['k']}",
                                    m*nr_max+r, nm_max*nr_max, c.config.log_substeps)

                    ##apply the increment to restart files (use io backend)
                    self.update_files(c, mem_id, rec_id)

        c.comm.Barrier()
        #c.comm.cleanup_file_locks()

    def prepare_async_file_io(self, c: Context, r: int, m: int, pid_active: bool):
        """
        Prepare file locks for asynchronous io, needed for blocking write (e.g. in netcdf without parallel support)
        """
        if c.config.io_mode != 'offline':
            return

        ##get file names for sync io
        mem_list = c.mem_list
        rec_list = c.state.rec_list
        if pid_active:
            mem_id = mem_list[c.pid_mem][m]
            rec_id = rec_list[c.pid_rec][r]
            rec = c.state.info.fields[rec_id].asdict()
            model = c.models[rec['model_src']]
            file = c.io.call_method(c, 'prior', getattr(model, 'filename'), member=mem_id, **rec)
        else:
            mem_id = 0
            rec_id = 0
            rec = {}
            file = None
        all_files = c.comm.allgather(file)

        ##create the file locks
        for file in all_files:
            c.comm.init_file_lock(file)

    @abstractmethod
    def compute_increment(self, c: Context) -> None:
        ...

    @abstractmethod
    def update_files(self, c: Context, mem_id: MemID, rec_id: FieldRecordID) -> None:
        ...