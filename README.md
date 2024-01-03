# NEDAS: The Next-generation Ensemble Data Assimilation System

NEDAS provides a light-weight Python solution to the ensemble data assimilation (DA) problem for geophysical models. It serves as a new test environment for DA researchers. Thanks to its modular and scalable design, new DA algorithms can be rapidly tested and developed even for large-dimensional models. NEDAS offers a collection of state-of-the-art DA algorithms, including serial assimilation approaches (similar to [DART](https://github.com/NCAR/DART) and [PSU EnKF](https://github.com/myying/PSU_WRF_EnKF) systems), and batch assimilation approaches (similar to the LETKF in [PDAF](https://pdaf.awi.de/trac/wiki), [JEDI](https://www.jcsda.org/jcsda-project-jedi), etc.). NEDAS offers DA researchers with new ideas to test/compare their methods/prototypes early-on in real-model-like environments, before committing resources to full implementation in operational systems.

# Table of Contents
[Quick Start Guide](#quick_start)

[Code Directories and Components](#code_structure)

[The DA Problem and Basic Design](#basic_design)

[Description of Key Variables and Functions](#descriptions)

[Adding New Models and Observations](#add_model_obs)

[Acknowledgements](#acknowledgements)


## Quick Start Guide <a name='quick_start'></a>

NEDAS is written in Python and Bash. To get started, you first need to setup your environment.

To get a copy of NEDAS, clone the repository in your `$CODE` directory:

`git clone git@github.com:nansencenter/NEDAS.git`

Install Python and the required libraries in `requirements.txt`, we recommend creating a separate environment for each project:

`python -m venv <my_python_env>`

Enter the environment by

`source <my_python_env>/bin/activiate`

and use a package manager to install the libraries, such as

`pip install -r requirements.txt`

To let Python find NEDAS modules, you need to add the directory to the Python search path. In your .bashrc (or other system configuration files), add the following line and then source:

`export PYTHONPATH=$PYTHONPATH:$CODE/NEDAS`

The runtime parameters for NEDAS are handled by the `config` module, it reads system environment variables through `os.environ`. A full list of config parameters is provided by `config/defaults`. For your experiment, make a copy of the config file and change parameters accordingly. In NEDAS, `$CONFIG_FILE` points to the config file you will use. We recommend you keep a separate config file for each experiment.

Before sourcing the `$CONFIG_FILE`, you need to mark the variables for export to the environment of subsequent commands (in bash it is done by `set -a`), so that the Python subprocesses will have access to those config parameters:

`set -a; source $CONFIG_FILE; set +a`

Then, you can go to the `scripts` directory to run the top-level control scripts:

`cd $CODE/NEDAS/scripts`

The bash scripts are typically submitted to a job scheduler on host supercomputers. In `config/env` you can create an initialization script for loading modules for the specific host machine. And in `scripts/job_submit.sh` make sure to provide the command for running jobs for this host machine. For example, in `config/defaults` the `env/betzy/base.src` is sourced, and we submit the job with

`sbatch run_cycle.sh`

Alternatively, the `run_cycle.sh` script can be run directly, if you are running the script on your local computer. For example, `config/env/localhost` sets up a local computer using `mpiexec`. Then, you run the job with

`./run_cycle.sh`

Results can then be found in the `$work_dir/$exp_name` directory.


## Code Directories and Components <a name='code_structure'></a>

* **requirements.txt** list the required Python libraries. We mainly use `numpy` for basic data structure and numerics, but note that your system might need the BLAS/LAPACK packages for `numpy` to achieve higher performance.

* **fft_lib.py** provides `fft2`,`ifft2` functions through the `pyFFTW` package, alternatively you can use `numpy.fft` if you don't have `FFTW` installed.

* **netcdf_lib.py** is a simple wrapper for `netCDF4` to read/write netcdf files, there is no parallel io support yet.

* **parallel.py** handles communication among MPI processes using `mpi4py`, and **log.py** provides functions to show runtime messages and progresses.

* **conversion.py** contains some unitily functions for unit and format conversion.

* **assim\_tools/** contains the functions handling model state variables in `state.py`, functions handling observations in `obs.py`, core DA algorithms in `analysis.py`, and post-processing functions in `update.py`.

* **grid/grid.py** provides a `Grid` class to handle the conversion between 2D fields.

* **models/** contains model modules (see details in its documentation), where users provide a set of functions `read_var`, `z_coords` etc. to interface with their forecast models.

* **dataset/** contains dataset modules (see details in its documentation), where users provide functions such as `read_obs` to pre-process with their dataset files and form the observation sequence.

* **perturb/** contains functions for generating random perturbations.

* **diag/** contains functions for computing misc. diagnostics for model forecast verification and filter performance evaluation.

* **config/** contains bash environment source files and configuration files, in Python program `import config as c` will load the configuration in `c` which will then be passed into functions.

* **scripts/** contains top-level bash control scripts such as `run_cycle.sh`, `run_forecast.sh` for a cycling DA experiment. Some model-specific scripts are located in their own `models/<model>/` directories, including initialisation scripts, run scripts, and post-processing scripts.

* **tutorials/** contains some Jupyter notebooks to illustrate how key functions work. Remember to enter your Python environment and `set -a; source $CONFIG_FILE`, before you start the notebook server. Note that all notebooks run in single processor mode.


## The DA Problem and Basic Design <a name='basic_design'></a>

DA seeks to optimally combine information from model forecasts and observations to obtain the best estimate of state/parameters of a dynamical system, which is called the analysis. The challenges in solving the analysis for modern geophysical models are: 1) the large-dimensional model state and observations, and 2) the nonlinearity in model dynamics and in state-observation relation. 

To address the first challenge, we employ distributed memory parallel computation strategy, since the entire ensemble state maybe too large to fit in the RAM of a single computer. And for the second challenge, we seek to test and compare new nonlinear DA methods (in the literature, or still in people's head) to try to tackle the problem.

A compromise is made in favor of code flexibility than its runtime efficiency. We aim for more modular design so that components in the DA algorithm can be easily changed/upgraded/compared. A pause-restart strategy is used: the model writes the state to restart files, then DA reads those files and computes the analysis and outputs to the updated files, and the model continues running. This is "offline" assimilation. In operational systems, sometimes we need "online" algorithms where everything is hold in the memory to avoid slow file I/O.  NEDAS provides parallel file I/O, not suitable for time-critical applications, but efficient enough for most research and development purposes.

The first challenge on dimensionality demands a careful design of memory layout among processors. The ensemble model state has dimensions: `member`, `variable`, `time`, `z`, `y`, `x`. When preparing the state, it is easier for a processor to obtain all the state variables for one member, since they are typically stored in the same model restart file. Each processor can hold a subset of the ensemble states, this memory layout is called "state-complete". To apply the ensemble DA algorithms, we need to transpose the memory layout to "ensemble-complete", where each processor holds the entire ensemble but only for part of the state variables ([Anderson & Collins 2007](https://doi.org/10.1175/JTECH2049.1)).

In NEDAS, for each member the model state is further divided into "fields" with dimensions (`y`,`x`) and "records" with dimensions (`variable`, `time`, `z`). Because, as the model dimension grows, even the entire state for one member maybe too big for one processor to hold in its memory. The smallest unit is now the 2D field, and each processor holds only a subset along the record dimension. Accordingly, the processors (`pid`) are divided into "member groups" (with same `pid_rec`) and "record groups" (with same `pid_mem`), see Fig. 1 for example. "State-complete" now becomes "field-complete". The record dimension allows parallel processing of different fields by the `read_var` functions in model modules. And during assimilation, each `pid_rec` only solves the analysis for its own list of `rec_id`.

| ![](https://github.com/nansencenter/NEDAS/blob/main/tutorials/imgs/transpose_state.png "Parallel memory layout for the state") |
|:---|
| **Figure 1**: Memory layout for 6 processors (`pid` = 0, ..., 5), divided into 2 member groups (`pid_rec` = 0, 1) and 3 record groups (`pid_mem` = 0, 1, 2). The state has dimensions: 100 members (`mem_id`), 16 partitions (`par_id`), and 50 records (`rec_id`). The field-complete **fields** hold all the partitions but only subset of the ensemble, after transpose (gray arrows), the ensemble-complete **state** holds all the members but only subset of partitions. |

For observations, it is easier to process the entire observing network at once, instead of going through the measurements one by one. Therefore, each observing network (record) is assigned a unique `obs_rec_id` to be handled by one processor.

Each `pid_rec` only need to process its own list of `obs_rec_id`, `pid_mem` = 0 is responsible for reading and processing the actual observations using `read_obs` functions from dataset modules, while all `pid_mem` separately process their own members for the observation priors.

As shown in Fig. 2, a transpose among different `pid_mem` brings the observations **obs** from field-complete to ensemble-complete, an additional transpose among different `pid_rec` gathers all `obs_rec_id` for each `rec_id` to form the final local observation **lobs**.

| ![](https://github.com/nansencenter/NEDAS/blob/main/tutorials/imgs/transpose_obs.png "Parallel memory layout for the observation") |
|:--|
| **Figure 2**: Same as Fig. 1 but showing the memory layout for observations. There are 5 observation records distributed over `pid_rec`. An additional transpose along the record dimension (yellow arrows) is necessary to allow observation records stored in other `pid_rec` to update the field records stored in each `pid_rec`. |

When the transpose is complete, on each `pid`, the local ensemble **state\_prior**[`mem_id`, `rec_id`][`par_id`] is updated to the posterior **state\_post**, using local observations **lobs**[`obs_rec_id`][`par_id`, `rec_id`] and observation priors **lobs\_prior**[`mem_id`, `obs_rec_id`][`par_id`, `rec_id`].

NEDAS provides two assimilation modes:

In batch mode, the analysis domain is divided into small local partitions (indexed by `par_id`) and each `pid_mem` solves the analysis for its own list of `par_id`. The local observations are those falling inside the localization radius for each [`par_id`,`rec_id`]. The "local analysis" for each state variable is computed using the matrix-version ensemble filtering equations (such as [LETKF](https://doi.org/10.1016/j.physd.2006.11.008), [DEnKF](https://doi.org/10.1111/j.1600-0870.2007.00299.x)). The batch mode is favorable when the local observation volume is small and the matrix solution allows more flexible error covariance modeling (e.g., to include correlations in observation errors).

In serial mode, we go through the observation sequence and assimilation one observation at a time. Each `pid` stores a subset of state variables and observations with `par_id`, here locality doesn't matter in storage, the `pid` owning the observation being assimilated will first compute observation-space increments, then broadcast them to all the `pid` with state\_prior and/or lobs\_prior within the observation's localization radius and they will be updated. For the next observation, the updated observation priors will be used for computing increments. The whole process iteratively updates the state variables on each `pid`. The serial mode is more scalable especially for inhomogeneous network where load balancing is difficult, or when local observation volume is large. The scalar update equations allow more flexible use of nonlinear filtering approaches (such as particle filter, rank regression).

NEDAS allows flexible modifications in the interface between model/dataset modules and the core assimilation algorithms, to achieve more sophisticated functionality:

Multiple time steps can be added in the `time` dimension for the state and/or observations to achieve ensemble smoothing instead of filtering. Iterative smoothers can also be formulated by running the analysis cycle as an outer-loop iteration (although they can be very costly).

Miscellaneous transform functions can be added for state and/or observations, for example, Gaussian anamorphosis to deal with non-Gaussian variables; spatial bandpass filtering to run assimilation for "scale components" in multiscale DA; neural networks to provide a nonlinear mapping between the state space and observation space, etc.


## Description of Key Variables and Functions <a name='descriptions'></a>

| ![](https://github.com/nansencenter/NEDAS/blob/main/tutorials/imgs/flowchart.png "Workflow for one assimilation cycle") |
|:--|
| **Figure 3**. Workflow for one assimilation cycle/iteration. For the sake of clarity, only the key variables and functions are shown. Black arrows show the flow of information through functions.|

Indices and lists:

* For each processor, its `pid` is the rank in the communicator `comm` with size `nproc`. The `comm` is split into `comm_mem` and `comm_rec`. Processors in `comm_mem` belongs to the same record group, with `pid_mem` in `[0:nproc_mem]`. Processors in `comm_rec` belongs to the same member group, with `pid_rec` in `[0:nproc_rec]`. Note that `nproc = nproc_mem * nproc_rec`, user should set `nproc` and `nproc_mem` in the config file.

* `mem_list`[`pid_mem`] is a list of members `mem_id` for processors with `pid_mem` to handle.

* `rec_list`[`pid_rec`] is a list of field records `rec_id` for processors with `pid_rec` to handle.

* `obs_rec_list`[`pid_rec`] is a list of observation records `obs_rec_id` for processors with `pid_rec` to handle.

* `partitions` is a list of tuples `(istart, iend, di, jstart, jend, dj)` defining the partitions of the 2D analysis domain, each partition holds a slice `[istart:iend:di, jstart:jend:dj]` of the field and is indexed by `par_id`.

* `par_list`[`pid_mem`] is a list of partition id `par_id` for processor with `pid_mem` to handle.

* `obs_inds`[`obs_rec_id`]['par'][`par_id`] is the indices in the entire observation record `obs_rec_id` that belong to the local observation sequence for partition `par_id`, and `obs_inds`[`obs_rec_id`]['rec'][`rec_id`], similarly, is the indices of local observations for the field record `rec_id`.


Data structures:

* `fields_prior`[`mem_id`, `rec_id`] points to the 2D fields `fld[...]` (np.array).

* `z_fields`[`mem_id`, `rec_id`] points to the z coordinate fields `z[...]` (np.array).

* `state_prior`[`mem_id`, `rec_id`][`par_id`] points to the field chunk `fld_chk` (np.array) in the partition.

* `obs_seq`[`obs_rec_id`] points to observation sequence `seq` that is a dictionary with keys ('obs', 't', 'z', 'y', 'x', 'err\_std') each pointing to a list containing the entire record.

* `lobs`[`obs_rec_id`][`par_id`, `rec_id`] points to local observation sequence `obs_chk` that is a dictionary with same keys as `seq` but the lists only contain a subset of the record.

* `obs_prior_seq`[`mem_id`, `obs_rec_id`] points to the observation prior sequence (np.array), same length with `seq['obs']`.

* `lobs_prior`[`mem_id`, `obs_rec_id`][`par_id`, `rec_id`] points to the local observation prior sequence (np.array), same length with `obs_chk`.


Functions:

* `prepare_state()`: For member `mem_id` in `mem_list` and field record `rec_id` in `rec_list`, load the model module and `read_var()` gets the variables in model native grid, convert to analysis grid, and apply miscellaneous user-defined transforms. Also, get z coordinates in the same prodcedure using `z_coords()` functions. Returns `fields_prior` and `z_fields`.

* `prepare_obs()`: For observation record `obs_rec_id` in `obs_rec_list`, load the dataset module and `read_obs()` get the observation sequence. Apply miscellaneous user-defined transforms if necessary. Returns `obs_seq`.

* `assign_obs()`: According to ('y', 'x') coordinates for `par_id` and ('variable', 'time', 'z') for `rec_id`, sort the full observation sequence `obs_rec_id` to find the indices that belongs to the local observation subset. Returns `obs_inds`.

* `prepare_obs_from_state()`: For member `mem_id` in `mem_list` and observation record `obs_rec_id` in `obs_rec_list`, compute the observation priors from model state. There are three ways to get the observation priors: 1) if the observed variable is one of the state variables, just get the variable with `read_field()`, or 2) if the observed variable can be provided by model module, then get it through `read_var()` and convert to analysis grid. These two options obtains observed variables defined on the analysis grid, then we convert them to the observing network and interpolate to the observed z location. Option 3) if the observation is a complex function of the state, the user can provide `obs_operator()` in the dataset module to compute the observation priors. Finally, the same miscellaneous user-defined transforms can be applied. Returns `obs_prior_seq`.

* `transpose_field_to_state()`: Transposes field-complete `field_prior` to ensemble-complete `state_prior` (illustrated in Fig.1). After assimilation, the reverse `transpose_state_to_field()` transposes `state_post` back to field-complete `fields_post`.

* `transpose_obs_to_lobs()`: Transpose the `obs_seq` and `obs_prior_seq` to their ensemble-complete counterparts `lobs` and `lobs_prior` (illustrated in Fig. 2).

* `batch_assim()`: Loop through the local state variables in `state_prior`, for each state variable, the local observation sequence is sorted based on the localization and impact factors. If the local observation sequence is not empty, compute the `local_analysis()` to update state variables, save to `state_post` and return.

* `serial_assim()`: Loop through the observation sequence, for each observation, the processor storing this observation will compute `obs_increment()` and broadcast. For all processors, if some of its local state/observations are within the localization radius of this observation, compute `update_local_ens()` to update these state/observations. Do this iteratively for all observations until end of sequence. Returns the updated local `state_post`.

* `update()`: Take `state_prior` and `state_post`, apply miscellaneous user-defined inverse transforms, compute `analysis_incr()`, convert the increment back to model native grid and add the increments to the model variables int he restart files. Apart from simply adding the increments, some other post-processing steps can be implemented, for example using the increments to compute optical flows and align the model variables instead.


## Adding New Models and Observations <a name='add_model_obs'></a>

To use NEDAS for your own models/observations, please read the detailed documentation for `models` and `dataset` modules, and create a module with functions to interface with the new models and/or dataset files. In the workflow chart the user-provided functions are highlighted in orange.

If you are considering DA experiments for a model, typically some Python diagnostic tools for the model state variables already exist, so the work for implementing the modules shall not be too heavy. Essentially you need to provide functions such as `read_var` to receive some key word arguments (variable name, time, member, vertical index, etc.) and return a 2D field containing the corresponding model state variable.

For observations, we expect you to already have some preprocessing scripts to read the raw dataset, quality control and screen for valid observations for the analysis domain, etc. These can be implemented in the `read_obs` function. Some steps in preprocessing are more involved: super-observation, uncertainty estimation, and extraction of information matching the model-resolved scales. We suggest you consult DA experts to implement these steps.

List of currently supported models and observations:

* The TOPAZ system coupled ocean (HYCOM) and sea ice (CICE4) model, with satellite obserations and insitu profiler data.

* The next-generation sea ice model (neXtSIM), with SAR-image-based sea ice drift and deformation observations.

and planned developement for:

* The Weather Research and Forecast (WRF) model (Polar WRF), with satellite observations.

* ECOSMO biogeochemistry model, with ocean color data.


### Acknowledgements <a name='acknowledgements'></a>

NEDAS was initiated by [Yue Ying](https://myying.github.io/) in 2022. Please cite this repository if you used NEDAS to produce results in your research publication/presentation.

The developement of this software was supported by the NERSC internal funding in 2022; and the Scale-Aware Sea Ice Project ([SASIP](https://sasip-climate.github.io/)) in 2023.

During the development, we received with contribution from: Anton Korosov, Timothy Williams (pynextsim libraries), NERSC-HYCOM-CICE group led by Annette Samuelsen (pythonlib for abfile, confmap, etc.), Jiping Xie (enkf-topaz), Tsuyoshi Wakamatsu (BIORAN), Francois Counillon, Yiguo Wang, Tarkeshwar Singh (EnOI, EnKF, and offline EnKS in NorCPM).

We provide the software "as is", the user is responsible for their own modification and ultimate interpretation of their research findings using the software. We welcome community feedback and contribution to support new models/observations, please use the "pull request" if you want to be part of the development effort.
