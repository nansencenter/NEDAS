# v1.1

    Use fully yaml-file-based configuration, no need for linux environment variables anymore.

    Model modules now become Model classes, with user provided functions (read_var, etc.). Two ensemble_forecast mode is used: batch run or using a scheduler.

    The entire workflow is based on python code now, scripts/run_exp.py is the top-level control script; assimilate.py and ensemble_forecast.py are the two main steps. More complex workflow can be introduced by the user, some examples will come in future releases.

    Some adaptive inflation algorithms added, more algorithms to be included soon.


# v1.0-beta

    Use bash scripts for workflow control, model code run using model/<model>/module_forecast.sh

    DA step is performed by scripts/run_assim.py in parallel

    Use environment variables defined in config/* to send config to individual programs

    Basic demo cases for the vort2d and qg models, a qg model benchmark is prepared for comparing efficiency of DA algorithms.

