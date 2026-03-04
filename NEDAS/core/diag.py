from .context import Context

class Diagnostics:
    """
    This class manages diagnostics functions
    """

    def __init__(self, ):
        pass


    def __call__(self, ):
        pass
        c.print_1p(f"Running diagnostics:")

        ##get task list for each rank
        task_list = bcast_by_root(c.comm)(self.distribute_diag_tasks)(c)

        ##the processor with most work load will show progress messages
        c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]

        ##init file locks for collective i/o
        self.init_file_locks(c)

        ntask = len(task_list[c.pid])
        for task_id, rec in enumerate(task_list[c.pid]):
            c.show_progress(f"PID {c.pid:4} running diagnostics '{rec['method']}'", task_id, ntask)

            method_name = f"NEDAS.diag.{rec['method']}"
            mod = importlib.import_module(method_name)

            ##perform the diag task
            mod.run(c, **rec)

        c.comm.Barrier()
        c.print_1p(' done.\n')
        c.comm.cleanup_file_locks()


    def distribute_diag_tasks(self, c):
        """Build the full task list and distribute among mpi ranks"""
        task_list_full = []
        for rec in ensure_list(c.diag):
            ##load the module for the given method
            method_name = f"NEDAS.diag.{rec['method']}"
            module = importlib.import_module(method_name)
            ##module returns a list of tasks to be done by each processor
            if not hasattr(module, 'get_task_list'):
                task_list_full.append(rec)
                continue
            task_list_rec = module.get_task_list(c, **rec)
            for task in task_list_rec:
                task_list_full.append(task)
        ##collected full list of tasks is evenly distributed across the mpi communicator
        task_list = distribute_tasks(c.comm, task_list_full)
        return task_list