import os
import inspect
from abc import ABC, abstractmethod
import numpy as np
from NEDAS.config import parse_config
from NEDAS.utils.progress import progress_bar

class Updator(ABC):
    """Base class for updators of the model restart files"""
    def __init__(self, c):
        ##get updator parameters from config file
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, parse_args=False, **c.updator_def)
        for key, value in config_dict.items():
            setattr(self, key, value)

        self.increment = {}

    def update(self, c, state):
        """
        Top-level routine to apply the analysis increments to the original model
        restart files (as initial conditions for the next forecast)

        Inputs:
        - c: config module
        - state: state object, where field_prior, field_post are
        the field-complete state variables before and after assimilate()
        """
        pid_mem_show = [p for p,lst in state.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in state.rec_list.items() if len(lst)>0][0]
        c.pid_show = pid_rec_show * c.nproc_mem + pid_mem_show

        c.print_1p(f'>>> update model restart files with analysis increments\n')

        ##compute analysis increments
        self.compute_increment(c, state)

        ##process the fields, each processor goes through its own subset of
        ##mem_id,rec_id simultaneously
        ##but need to keep every rank in sync to coordinate multiprocess file access
        nm_max = np.max([len(lst) for _,lst in state.mem_list.items()])
        nr_max = np.max([len(lst) for _,lst in state.rec_list.items()])
        for r in range(nr_max):
            for m in range(nm_max):
                ##get file names for sync io
                pid_active = ( m < len(state.mem_list[c.pid_mem]) and r < len(state.rec_list[c.pid_rec]) )
                if pid_active:
                    mem_id = state.mem_list[c.pid_mem][m]
                    rec_id = state.rec_list[c.pid_rec][r]
                    rec = state.info['fields'][rec_id]
                    path = c.forecast_dir(rec['time'], rec['model_src'])
                    model = c.models[rec['model_src']]
                    file = model.filename(path=path, member=mem_id, **rec)
                else:
                    file = None
                all_files = c.comm.allgather(file)
                ##create the file locks
                for file in all_files:
                    c.comm.init_file_lock(file)

                if pid_active:
                    if c.debug:
                        print(f"PID {c.pid:4}: update_restartfile mem{mem_id+1:03} '{rec['name']:20}' {rec['time']} k={rec['k']}", flush=True)
                    else:
                        c.print_1p(progress_bar(m*nr_max+r, nm_max*nr_max))

                    ##apply the increment to restart files
                    self.update_restartfile(c, state, mem_id, rec_id)

        c.comm.Barrier()
        #c.comm.cleanup_file_locks()
        c.print_1p(' done.\n')

    @abstractmethod
    def compute_increment(self, c, state):
        pass

    @abstractmethod
    def update_restartfile(self, c, state, mem_id, rec_id):
        pass