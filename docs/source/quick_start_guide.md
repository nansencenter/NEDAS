# Quick Start Guide

- Create a python environment for your experiment (optional but recommended)

    Install Python then create environment named `<my_python_env>`

    `python -m venv <my_python_env>`

    Enter the environment by

    `source <my_python_env>/bin/activiate`

- Make a copy of the NEDAS code and place it in your `<code_dir>`

    `cd <code_dir>`

    `git clone git@github.com:nansencenter/NEDAS.git`

- Install the required libraries, as listed in `requirements.txt`

    Using a package manager such as pip, you can install them by

    `pip install -r requirements.txt`

- Add NEDAS directory to the Python search path

    To let Python find NEDAS modules, you can add the NEDAS directory to the search paths. In your .bashrc (or other system configuration files), add the following line and then source:

    `export PYTHONPATH=$PYTHONPATH:<code_dir>/NEDAS`

- Make the yaml configuration file for your experiment

    A full list of configuration variables and their default values are stored in `config/default.yml`. There are sample configuration files in `config/samples/*`, you can make a copy to `<my_config_file>` and make changes.

- Setup runtime environment for the host machine

    In `<my_config_file>`:

    Set `work_dir` to the working directory for the experiment.

    Set `job_submit_cmd` to the parallel job submit command/script on the host machine, see example `config/samples/job_submit_betzy.sh` for more details.

    Set `nproc` to the number of processors to be used for the experiment.

- Setup models and datasets

    In `models/<model_name>`, edit `setup.src` to provide environment for running the model. `model_code_dir` is where the model code is; `model_data_dir` is where the static input files are that the model requires during runtime; `ens_init_dir` is where the initial restart files are for the first cycle of the experiment.

    When you are trying out NEDAS for the first time, you can start from the `vort2d` model (written in Python), its setup is easy and `vort2d.yml` is a sample config file. The `qg` model is another toy model, it is written in Fortran and requires installation, it is a good next step to get to know the details of NEDAS and working towards adding your own model class.

    For the datasets that provide observations to be assimilated, setup their directories in config file, and make sure you implemented the `dataset.<dataset_name>` module.

- Start the experiment

    In `tutorials` there are some jupyter notebooks to demonstrate the DA workflow for some supported models.

    On <my_host_machine>, you can start a notebook by

    `jupyter-notebook --ip=0.0.0.0 --no-browser --port=<port>`

    then create another ssh connection to the machine

    `ssh -L <port>:localhost:<port> <my_host_machine>`

    once the connection is established, you can access the notebook from your local browser via `localhost:<port>/tree?`

    In jupyter notebooks you can quickly check the status of model states, observations, and diagnosing the DA performance, you can play with the DA workflow, modify it and create your own approach.

    Once you finished debugging and are happy with the new workflow, you can run the experiments without the jupyter notebooks. In `scripts` the `run_expt.py` gives an example of the top-level control workflow to perform cycling DA experiments. Run the experiment by `python run_expt.py --config_file=<my_config_file>`

    On betzy, the `sbatch submit_job.sh` command submits a run to the job queue, so that many experiments can be run simultaneously.

