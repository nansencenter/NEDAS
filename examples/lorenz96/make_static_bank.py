"""
Build a bank of static ensemble members for the hybrid covariance (covariance_def) from a long
free run of the Lorenz-96 model: after a spin-up, one state every `interval` hours is saved as a
restart file at its own time in static_dir, and listed in static_list.txt as '<time>'.

usage: python make_static_bank.py static_dir nens_static [--interval 72] [--spinup 50]
                                  [--time_start 2000-01-01T00:00:00] [--seed 0]
"""
import os
import argparse
from datetime import datetime, timedelta, timezone
import numpy as np
from NEDAS.models.lorenz96.lorenz96_model import Lorenz96Model

def make_static_bank(static_dir: str, nens_static: int, interval: float=72, spinup: float=50,
                     time_start: datetime=datetime(2000, 1, 1, tzinfo=timezone.utc), seed: int=0) -> str:
    """
    Write nens_static states of a free run (spinup in model time units, interval in hours)
    to static_dir and return the static_list file
    """
    model = Lorenz96Model(io_mode='offline')
    os.makedirs(static_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    state = model.advance_time(rng.normal(0, 1, model.nx), spinup)  # spin-up onto the attractor
    static_list = os.path.join(static_dir, 'static_list.txt')
    time = time_start
    with open(static_list, 'w') as f:
        for _ in range(nens_static):
            model.write_var(state, name='state', member=None, time=time, path=static_dir)
            f.write(f"{time:%Y-%m-%dT%H:%M:%S}\n")
            state = model.advance_time(state, interval / model.hours_per_unit_time)
            time += timedelta(hours=interval)
    return static_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('static_dir')
    parser.add_argument('nens_static', type=int)
    parser.add_argument('--interval', type=float, default=72, help='hours between samples')
    parser.add_argument('--spinup', type=float, default=50, help='spin-up length in model time units')
    parser.add_argument('--time_start', default='2000-01-01T00:00:00')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    time_start = datetime.fromisoformat(args.time_start).replace(tzinfo=timezone.utc)
    print(make_static_bank(args.static_dir, args.nens_static, args.interval, args.spinup, time_start, args.seed))
