# Olivia Example

This is a simple multiprocess example, cycling for ~2 weeks including a spinup
period and a 3 day forecast at the end.
Minimal settings include:

* 16 processes (despite requesting 64 tasks from SLURM)
* 16 ensemble members

Should take ~10 minutes.

Note that one can bind a local version of NEDAS into the container by instead
using the command:

```
apptainer exec \
    --bind /path/to/local/NEDAS:/home/appuser/NEDAS \
    $container_path \
    python -m NEDAS --config_file=$config_file
```

note the extra `--bind` line.
