Configuration file
==================

Usage
-----

NEDAS configuration is driven by YAML files and runtime argument parsing.
The ``NEDAS/config/default.yml`` file defines all the entries and their default values.
At runtime, a customized configuration file can be used by ``-c CONFIG_FILE``,
the ``CONFIG_FILE`` doesn't need to define every entry in ``default.yml``,
just the ones related to the particular experiment.
Also, the simple entry types (not the compound types such as list, tuple and dict) can be
specified with a new value with ``--key value`` at runtime,
which makes it easier to run the same experiment but just changing one or two parameters in the configuration.

To run a NEDAS experiment on command line:

.. code-block:: bash

   python -m NEDAS -c CONFIG_FILE --key value


Alternatively, in an interactive environment such as a Jupyter notebook,
the configuration object ``config`` can be initialized directly with

.. code-block:: python

   from NEDAS.config import Config
   config = Config(config_file='CONFIG_FILE', key=value)

The ``config`` object can then use used to initialize and run the analysis scheme

.. code-block:: python

   from NEDAS.schemes import get_scheme
   scheme = get_scheme(config)
   scheme()

Description of entries
----------------------

System paths and runtime environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Entry
     - Description
     - Default (from ``NEDAS/config/default.yml``)
   * - ``work_dir``
     - Working directory for running the analysis scheme.
     - 'work'
   * - ``directories``
     - Runtime directory structure defined by format strings.
     - See details in Table 1.
   * - ``python_env``
     - Initialization script to enter the python environment.

       If not None, at runtime ``". {python_env}"`` will source

       this script before running the python command.
     - None
   * - ``io_mode``
     - I/O mode.

       ``'online'`` keeps model/dataset data in memory;
       
       ``'offline'`` uses files on disk.
     - 'offline'
   * - ``job_submit``
     - Runtime job submitter settings.

       These options are forwarded to the job submitter
       
       (see :doc:`NEDAS.job_submitters`).
     - See details in Table 2.
   * - ``nproc``
     - Total number of processors used when a step is
     
       executed under MPI.
     - 1
   * - ``nproc_mem``
     - Number of processors in a "member group" when 
     
       distributing ensemble members.

       If not set in YAML, the code sets ``nproc_mem = nproc``
       
       and computes ``nproc_rec = nproc / nproc_mem``.

       Must evenly divide ``nproc``.
     - None
     
       (interpreted as ``nproc``)
   * - ``nproc_util``
     - Number of processors to use for utility steps (preprocess,
     
       postprocess, diagnose, etc.).

       If not set in YAML, the code uses ``nproc_util = nproc``.
     - None

.. list-table:: Table 1. Breakdown of ``directories`` dictionary
   :header-rows: 1

   * - Key
     - Description
     - Default
   * - ``cycle_dir``
     - Directory for each analysis cycle.
     - | '{work_dir}/cycle/
       | {time:%Y%m%d%H%M}'
   * - ``forecast_dir``
     - Directory for the ensemble forecast step.
     - | '{work_dir}/cycle/
       | {time:%Y%m%d%H%M}/
       | {model_name}'
   * - ``analysis_dir``
     - Directory for the assimilation step
     
       (outer-loop iteration ``iter`` is part of the path).
     - | '{work_dir}/cycle/
       | {time:%Y%m%d%H%M}/
       | analysis/{iter}'

.. list-table:: Table 2. Breakdown of ``job_submit`` dictionary.
   :header-rows: 1

   * - Key
     - Description
     - Examples
   * - ``host``
     - Host machine type.

       Machine-specific behavior can be defined 
       
       in the corresponding subclass
       
       in :doc:`NEDAS.job_submitters`.
     - 'local', 'betzy', ...
   * - ``scheduler``
     - Scheduler type.
     - None, 'slurm', 'oar', 'pbs', ...
   * - ``project``
     - Project number for resource allocation.
     - None, 'nn2993k', ...
   * - ``queue``
     - Name of the scheduler queue to
     
       submit jobs to (HPC).
     - None, 'normal', 'devel', ...
   * - ``parallel_mode``
     - Parallelization strategy to request
     
       from the job submitter.
     - 'serial', 'mpi', 'openmp'

Runtime logging
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Entry
     - Description
     - Default
   * - ``debug``
     - If True, show extra debug messages and output intermediate
     
       data during runtime.
     - False
   * - ``timer``
     - If True, show elapsed time for major steps in the workflow.
     - True
   * - ``interactive``
     - If True, allow ANSI escape codes (colors, cursor movement)
     
       in terminal output.

       If None, auto-detected from the terminal environment.
     - True
   * - ``quiet``
     - If True, suppress most runtime status output.
     - False
   * - ``call_stack``
     - Current call stack context string, set automatically
     
       at runtime.
     - None
   * - ``call_stack_max_level``
     - Maximum call stack depth to display in status output.

       If None, all levels are shown.
     - 2
   * - ``is_notebook``
     - If True, adapt output formatting for Jupyter notebooks.

       If None, auto-detected.
     - None
   * - ``cols``
     - Terminal width in characters used for formatting status lines.

       If None, auto-detected from the terminal.
     - None
   * - ``anchor``
     - Number of characters reserved for the left (description)
     
       part of status lines.
     - 50
   * - ``tabspace``
     - Number of spaces per call stack level indentation.
     - 4
   * - ``progress_bar_width``
     - Width of the progress bar in characters.
     - 10

Analysis scheme design parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Default
   * - ``nens``
     - Ensemble size.
     - 20
   * - ``run_preproc``
     - Whether to run the preprocessing step.
     - True
   * - ``run_forecast``
     - Whether to run the ensemble forecast step.
     - True
   * - ``run_analysis``
     - Whether to run the analysis (assimilation) step.
     - True
   * - ``run_postproc``
     - Whether to run the postprocessing step after assimilation.
     - True
   * - ``run_diagnose``
     - Whether to run the diagnostic tools.
     - True
   * - ``save_checkpoint``
     - If True, save checkpoints of model state and observations
     
       between cycles.
     - False
   * - ``step``
     - Used by :mod:`NEDAS.schemes.filter`.

       If None, will run the entire workflow.

       Otherwise, will only run the specified step.

       (Valid step names depend on the scheme; for the filter scheme
       
       these include ``run_all``, ``prepare_truth``, 
       
       ``prepare_init_ensemble``, ``preprocess``, ``perturb``,
       
       ``filter``, ``postprocess``, ``ensemble_forecast``,
       
       and ``diagnose``.)
     - None

Time controls
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``time_start``
     - Start time of the period of interest.
     - 2001-01-01T00:00:00Z
   * - ``time_end``
     - End time of the period of interest.
     - 2001-01-30T00:00:00Z
   * - ``time_analysis_start``
     - Time of the first analysis cycle.

       Defaults to ``time_start`` if not set.
     - 2001-01-07T00:00:00Z
   * - ``time_analysis_end``
     - Time of the last analysis cycle.

       Defaults to ``time_end`` if not set.
     - 2001-01-28T00:00:00Z
   * - ``cycle_period``
     - Interval in hours between analysis cycles.
     - 12
   * - ``time``
     - Time of the current analysis cycle,
     
       set automatically at runtime.

       If None, will start at ``time_start``.
     - None
   * - ``obs_time_steps``
     - Time steps in hours relative to the analysis

       for the observations.
     - [0]
   * - ``obs_time_scale``
     - Smoothing window in hours for observations.
     - 0
   * - ``state_time_steps``
     - Time steps in hours relative to the analysis

       for the state variables.
     - [0]
   * - ``state_time_scale``
     - Smoothing window in hours for state variables.
     - 0

Analysis grid definition
^^^^^^^^^^^^^^^^^^^^^^^^

The ``grid_def`` entry is a dictionary with the following entries:

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``type``
     - Type of grid to use for the analysis step.

       If 'custom', the other entries will be used as kwargs in

       initializing a regular grid, see details in Table 3.

       If a model name is specified, the corresponding

       model grid will be used instead.
     - 'custom', 'qg', etc.
   * - ``mask``
     - Mask for invalid points in the domain.

       If not None, the model name specifies which model generates

       the mask for the analysis grid.
     - None, 'qg', etc.

.. list-table:: Table 3. Additional kwargs for custom regular grid generation.
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``proj``
     - Map projection defined as PROJ4 strings
     - None,

       '+proj=stere +lat_0=90 +lon_0=-45'
   * - ``xmin``
     - X coordinate start
     - 0
   * - ``xmax``
     - X coordinate end
     - 128
   * - ``ymin``
     - Y coordinate start
     - 0
   * - ``ymax``
     - Y coordinate end
     - 128
   * - ``dx``
     - Grid spacing

       Note: the coordinates and grid spacing

       should be in meters. But if proj is None,

       they can be nondimensional.
     - 1
   * - ``centered``
     - If True, the coordinates are defined

       at the center of each grid box.
     - False
   * - ``cyclic_dim``
     - The dimension(s) that are cyclic
     - None, 'x', 'y', or 'xy'
   * - ``distance_type``
     - Type of distance function
     - 'cartesian' or 'spherical'

State definition
^^^^^^^^^^^^^^^^

The ``state_def`` entry is a list, each item is a dictionary that defines one model state variable:

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``name``
     - Model state variable name.

       Corresponding to the keys in ``Model.variables``

       implemented in the model interface
     - 'streamfunc'
   * - ``model_src``
     - Name of the model this variable comes from.

       Should be one of the keys in ``model_def``.
     - 'qg.fortran'
   * - ``var_type``
     - Variable type.
     - 'field', or 'scalar'
   * - ``err_type``
     - Error distribution type.
     - 'normal'

The ``model_def`` entry is a dictionary, with model_name as keys pointing to a dictionary of model-specific configuration parameters.

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``config_file``
     - YAML configuration file for the model.

       If not specified, will use ``default.yml``

       in the corresponding model module directory.

       Additional entries added below will overwrite

       the settings in the YAML file, making it easier

       to setup twin experiments.
     - None,

       | '{nedas_root}/models/qg/
       | fortran/default.yml'
   * - ``model_env``
     - Initialization script for model.

       At runtime ``". {model_env}"`` will source

       this script before running the model forecast.
     - 'setup.src'
   * - ``model_code_dir``
     - Path to the model code directory.
     - '{nedas_root}/models/qg/fortran'
   * - ``nproc_per_run``
     - Number of processors to use for a model forecast.
     - 1
   * - ``nproc_per_util``
     - Number of processors to use for utility functions.
     - 1
   * - ``walltime``
     - Maximum runtime in seconds for model forecast.
     - 3600
   * - ``restart_dt``
     - Model restart file saving interval in hours.
     - 24
   * - ``forcing_dt``
     - Model boundary condition interval in hours.
     - 24
   * - ``ens_run_strategy``
     - Strategy for running tasks involving
     
       an ensemble of tasks.

       'scheduler': run each member as a separate job

       and distribute the workload using a :class:`Scheduler`.

       'batch': run all members in a single job.
     - 'scheduler' or 'batch'
   * - ``use_job_array``
     - Whether to use job array functionality when

       submitting the jobs via :class:`JobSubmitter`.
     - False
   * - ``ens_init_dir``
     - Directory where the initial ensemble restart files

       are located.
     - '{work_dir}/init_ens'
   * - ``truth_dir``
     - Directory where the truth files are located.

       This is required when using synthetic observations 
       
       that are generated from a truth run.
     - '{work_dir}/truth'

Observation definition
^^^^^^^^^^^^^^^^^^^^^^

The ``obs_def`` entry is a list, each item is a dictionary that defines one observation variable:

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``name``
     - Observation variable name.

       Corresponding to the keys in ``Dataset.variables``

       implemented in the dataset interface
     - 'velocity'
   * - ``dataset_src``
     - Name of the dataset the observation comes from.

       Should be one of the keys in ``dataset_def``.
     - 'synthetic'
   * - ``model_src``
     - Name of the model from which to compute the

       observation priors.
     - 'qg.fortran'
   * - ``nobs``
     - Number of observations.

       If generating synthetic random observation network,

       use this to control the density.
     - 3000
   * - ``err``
     - Error definition dictionary.
     - See details in Table 4
   * - ``hroi``
     - Horizontal localization distance,

       radius of influence beyond which the observation

       impact is tapered to zero.

       In the same units as grid coordinates
     - inf, 10, etc.
   * - ``vroi``
     - Vertical localization distance,

       in the same units as ``z_coords``
     - inf
   * - ``troi``
     - Temporal localization distance
     - inf
   * - ``impact_on_variable``
     - List of impact factors of this observation

       on the state/obs variables.

       The unlisted variable has a default impact of 1.0
     - { 'streamfunc': 0 },

       which turns off the

       impact on streamfunc

.. list-table:: Table 4. Breakdown of the observation error definition dictionary.
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``type``
     - Type of error distribution.
     - 'normal'
   * - ``std``
     - Observation error standard deviation.
     - 1.0
   * - ``hcorr``
     - Horizontal correlation length in observation error.
     - 0
   * - ``vcorr``
     - Vertical correlation length in observation error.
     - 0
   * - ``tcorr``
     - Temporal correlation length in observation error.
     - 0
   * - ``cross_corr``
     - Cross-variable correlation in observation error. A dictionary

       {variable_name: corr} listing the correlation between self

       and other variable_name. Auto-correlation is always 1,

       so there is no need to include self in the dictionary.
     - {'streamfunc': 0}

The ``dataset_def`` entry is a dictionary, with dataset_name as keys pointing to a dictionary of dataset-specific configuration parameters.

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``model_src``
     - Name of the model used for computing
     
       observation priors for this dataset.

       Should be one of the keys in ``model_def``.
     - 'qg.fortran'
   * - ``config_file``
     - YAML configuration file for the dataset.

       If not specified, will use ``default.yml``

       in the corresponding dataset module directory.

       Additional entries added below will overwrite

       the settings in the YAML file.
     - None
   * - ``dataset_dir``
     - Path to the dataset files.

       (For synthetic observations this can be left empty.)
     - None
   * - ``obs_window_min``
     - Start of the observation window, hours relative to 
     
       the analysis time.
     - -6
   * - ``obs_window_max``
     - End of the observation window, hours relative to 
     
       the analysis time.
     - 0

Some additional parameters:

Synthetic observations are enabled by using a synthetic dataset in ``obs_def`` (e.g. ``dataset_src: synthetic``)
and providing a corresponding entry in ``dataset_def``.

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Default (from ``NEDAS/config/default.yml``)
   * - ``shuffle_obs``
     - Whether to randomize the order of observations.
     - False
   * - ``z_coords_from``
     - Where the reference vertical coordinates come from.
     - 'mean'
   * - ``interp_method``
     - Interpolation method used when mapping between grids.
     - 'linear'

Perturbation
^^^^^^^^^^^^

The top-level ``perturb`` entry controls the optional perturbation step.
In the default configuration it is left empty/None (no perturbation).

If enabled, ``perturb`` should be a list of dictionaries. Each dictionary defines a perturbation to apply
to one ensemble member and one or more variables (see :mod:`NEDAS.core.perturb`).

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``variable``
     - Variable name (string) or list of variable names 
     
       to perturb.
     - 'streamfunc'
   * - ``model_src``
     - Model name the variable(s) come from 
     
       (a key in ``model_def``).
     - 'qg.fortran'
   * - ``type``
     - Perturbation type string.

       The first token selects the main method: 
       
       ``gaussian``, ``powerlaw``, or ``displace``.

       Additional options can be appended with commas 
       
       (e.g. ``gaussian,exp``).
     - 'gaussian'
   * - ``amp``
     - Perturbation amplitude.
     - 0.1
   * - ``hcorr``
     - Horizontal correlation length
     
       (needed by ``gaussian`` and ``displace``).
     - 15
   * - ``tcorr``
     - Temporal correlation length (hours) used

       to correlate perturbations between cycles/time steps.
     - 0
   * - ``powerlaw``
     - Power-law exponent
     
       (needed by ``powerlaw`` perturbations).
     - 4
   * - ``bounds``
     - Optional value bounds ``[vmin, vmax]`` enforced
     
       after perturbation.
     - [0, inf]
   * - ``seed``
     - Optional random seed.
     - 1234

If no perturbation is needed, leave ``perturb`` empty/None.

Assimilation method
^^^^^^^^^^^^^^^^^^^

The following parameters help NEDAS locate the correct analysis scheme and assimilation components.

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Default / example
   * - ``scheme``
     - Type of analysis scheme to use.
     - 'filter'
   * - ``assimilator_def``
     - Assimilator configuration dictionary.

       The assimilator class is chosen by
       
       ``assimilator_def.type``.
     - See below.
   * - ``updator_def``
     - Updator configuration dictionary
     
       (applies increments to produce posterior state).

       Alignment-based updators are selected via 
       
       ``updator_def.type`` and further configured
       
       through ``updator_def.config_file``.
     - See ``NEDAS/config/default.yml``
   * - ``covariance_def``
     - Covariance configuration dictionary.
     - See ``NEDAS/config/default.yml``

.. list-table:: Breakdown of ``assimilator_def``.
   :header-rows: 1

   * - Key
     - Description
     - Default
   * - ``type``
     - Assimilator type.
     - 'ETKF'
   * - ``config_file``
     - Optional YAML configuration file for the assimilator.

       If not specified, the assimilator module default is used.
     - None

Covariance inflation parameters are stored in the ``inflation_def`` entry as a dictionary.

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``type``
     - Type of inflation (post/prior, multiplicative/RTPP).
     - 'post,multiplicative'
   * - ``adaptive``
     - Whether to run an adaptive inflation scheme.
     - False
   * - ``coef``
     - Static inflation coefficient.
     - 1.0
   * - ``post_infl_formula``
     - Only used by adaptive posterior ``multiplicative`` inflation. Which Desroziers-derived
       estimator to use for the inflation ratio: ``omaamb`` (default, :math:`\langle(a-b)(o-a)\rangle/\mathrm{vara}`)
       matches Ying (2019)'s original formula; ``omb2_amb2`` (:math:`(\mathrm{omb}^2-\mathrm{varo}-\mathrm{amb}^2)/\mathrm{vara}`)
       is an alternative estimator derived under a different independence assumption.
     - 'omaamb'

Covariance localization settings are separately defined for the spatial and temporal components.
The ``localization_def`` entry is a dictionary with keys ``horizontal``, ``vertical`` and ``temporal``
each pointing to a dictionary that defines its localization function parameters.

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``type``
     - Type of localization kernel to use.

       Implemented types include ``gaspari_cohn``,
       
       ``step``, and ``exponential``.
     - 'gaspari_cohn'

State and observation transforms can be configured with the ``transform_def`` entry,
which is a list of dictionaries each defining one transform to apply
(see :mod:`NEDAS.assim_tools.transforms`).

.. list-table:: Breakdown of a ``transform_def`` entry.
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``type``
     - Transform type.

       Built-in types include ``scale_bandpass`` (for multiscale DA)
       
       and ``identity``.
     - 'scale_bandpass'
   * - ``decompose_obs``
     - If True, apply the same transform decomposition to 
     
       observations as well as state variables.
     - False

Multiscale approach configuration:

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``niter``
     - Number of outer-loop iterations, e.g. number of 
     
       scale components in a multiscale approach.
     - 1
   * - ``iter``
     - Current iteration number
     - 0
   * - ``resolution_level``
     - Resolution level (n) for the analysis grid.

       The analysis grid will have a resolution ``dx * 2**n``

       where ``dx`` is the grid spacing defined in ``grid_def``.
     - [0]
   * - ``character_length``
     - Characteristic length (in grid coordinate units) 
     
       for each scale (large to small).
     - [16]
   * - ``localize_scale_fac``
     - Scale factor for localization distances.
     - [1]
   * - ``obs_err_scale_fac``
     - Scale factor for observation error inflation.
     - [1]

Diagnostic methods
^^^^^^^^^^^^^^^^^^

The ``diag`` entry is a list. Each element is a dictionary defining a diagnostic method to be run.

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - Example
   * - ``method``
     - Name of the diagnostic method 
     
       (Python module path under ``NEDAS/diag``).
     - 'misc.convert_output'
   * - ``config_file``
     - Optional YAML configuration file
     
       for the method.

       If not specified, the method module
       
       default is used.
     - None
   * - ``model_src``
     - Which model the diagnostic is applied to.
     - 'qg.fortran'
   * - ``variables``
     - List of variables to process.
     - ['streamfunc']
   * - ``grid_def``
     - Optional output grid definition; 
     
       if omitted, the model grid is used.
     - None
   * - ``file``
     - Output filename format string.
     - | '{work_dir}/output/
       | mem{member:03}_
       | {time:%Y-%m-%dT%H}.nc'
