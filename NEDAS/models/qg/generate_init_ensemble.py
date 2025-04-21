import os
from datetime import timedelta
from NEDAS.config import Config
from NEDAS.schemes.offline_filter import OfflineFilterAnalysisScheme

scheme = OfflineFilterAnalysisScheme()
c = Config(parse_args=True)
model = c.model_config['qg']

print(f"Creating initial condition for qg model:")
init_time = c.time
c.time -= model.spinup_hours * timedelta(hours=1)
c.cycle_period = model.spinup_hours
print(f"initial condition type: {model.psi_init_type}")
print(f"spinup period: {model.spinup_hours} hours")

print(f"Running ensemble forecast from {c.time} to {c.next_time}")
scheme.run_step(c, 'ensemble_forecast')

print("Moving output files")
fcst_dir = c.forecast_dir(c.time, 'qg')
cycle_dir = c.cycle_dir(c.time)
ens_init_dir = model.ens_init_dir
basename = f"output_{init_time:%Y%m%d_%H}.bin"
for m in range(c.nens):
    mstr = f"{m+1:04d}"

    src_file = os.path.join(fcst_dir, mstr, basename)
    init_file = os.path.join(ens_init_dir, mstr, basename)

    os.system(f"mkdir -p {os.path.dirname(init_file)}")
    os.system(f"mv -v {src_file} {init_file}")

print(f"removing temporary run directory: {cycle_dir}")
os.system(f"rm -rf {cycle_dir}")
