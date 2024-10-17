import numpy as np
import os
from utils.conversion import dt1h, ensure_list
from utils.parallel import distribute_tasks, bcast_by_root, by_rank
from utils.progress import timer, print_with_cache, progress_bar
from utils.dir_def import forecast_dir, analysis_dir

diag_script_path = os.path.abspath(__file__)

def diag(c):
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)
    print_1p('\nRunning diagnostics \n')


if __name__ == "__main__":
    from config import Config
    c = Config(parse_args=True)  ##get config from runtime args

    timer(c)(diag)(c)

