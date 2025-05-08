Configuration file
==================

.. contents::
   :local:
   :depth: 2

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

In a python script, the following code can be included

.. code-block:: python

   from NEDAS.config import Config
   c = Config(parse_args=True)

so that when the script is run on command line as

.. code-block:: bash

   python script.py -c CONFIG_FILE --key value

the config object ``c`` is created, whose attributes carry the configuration parameters.

Alternatively, in an interactive environment such as a Jupyter notebook,
the configuration object ``c`` can be initialized directly with

.. code-block:: python

   from NEDAS.config import Config
   c = Config(config_file='CONFIG_FILE', key=value)

Description of entries
----------------------
System paths and runtime environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Entry
     - Description
     - Default
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
   * - ``job_submit``
     - Runtime job submitter settings, which are passed to
     
       :func:`NEDAS.utils.shell_utils.run_job` as kwargs.
     - See details in Table 2.
   * - ``nproc``
     - Number of processors to use for the analysis step.
     - 10
   * - ``nproc_mem``
     - Number of processors in a "member group",

       which splits the MPI communicator ``comm`` of size ``nproc``
       
       into ``comm_mem`` of size ``nproc_mem``

       If None, no splitting is done.
     - None
   * - ``nproc_util``
     - Number of processors to use for utility steps,

       such as preproc, postproc, diagnose, etc.

       If None, will use the same as ``nproc``.
     - None

.. list-table:: Table 1. Breakdown of ``directories`` dictionary
   :header-rows: 1
   :widths: 20 30 50

   * - Key
     - Description
     - Default
   * - ``cycle_dir``
     - Directory for each

       analysis cycle
     - '{work_dir}/cycle/{time:%Y%m%d%H%M}'
   * - ``analysis_dir``
     - Directory for the

       assimilation step
     - '{work_dir}/cycle/{time:%Y%m%d%H%M}/analysis'
   * - ``forecast_dir``
     - Directory for the

       ensemble forecast step
     - '{work_dir}/cycle/{time:%Y%m%d%H%M}/{model_name}'

.. list-table:: Table 2. Breakdown of ``job_submit`` dictionary.
   :header-rows: 1
   :widths: 20 30 50

   * - Key
     - Description
     - Examples
   * - ``host``
     - Host machine name, machine-specific behavior

       in job scheduling can be defined in the
       
       corresponding subclass in :doc:`NEDAS.job_submitters`.
     - None, 'laptop', 'betzy', etc.
   * - ``project``
     - Project number for resource allocation
     - None, 'nn2993k', etc.
   * - ``queue``
     - Name of the scheduler queue to submit jobs to
     - None, 'normal', 'devel', etc.
   * - ``scheduler``
     - Scheduler type.
     
       Typically a separate :doc:`NEDAS.job_submitters`

       subclass is defined for each scheduler type.
     - None, 'slurm', 'oar', 'pbs', etc.
   * - ``ppn``
     - Number of available processors

       per compute node
     - 128
   * - ``run_separate_jobs``
     - Whether the jobs will be submitted separately 

       to the scheduler, or just run as steps in a
       
       shared job allocation.
     - False

Analysis scheme design parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Key
     - Description
     - Default
   * - ``nens``
     - Ensemble size
     - 20
   * - ``run_preproc``
     - Whether to run the preprocessing step.
     - True
   * - ``run_forecast``
     - Whether to run the ensemble forecast step.
     - True
   * - ``run_analysis``
     - Whether to run the analysis step.
     - True
   * - ``run_diagnose``
     - Whether to run the diagnostic tools.
     - True
   * - ``debug``
     - If True, show extra debug message and output

       intermediate data during runtime.
     - False
   * - ``timer``
     - If True, show elapsed time for each

       major steps in the workflow.
     - True
   * - ``step``
     - Used by :mod:`NEDAS.schemes.offline_filter`.

       If None, will run the entire workflow.

       Otherwise, will only run the specified step

       defined in the workflow at ``time``.
     - None, 'preprocess', 'postprocess',

       'filter', 'perturb', 'diagnose',

       or 'ensemble_forecast'.

Time controls
^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1
   :widths: 20 45 35

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
     - 2001-01-07T00:00:00Z
   * - ``time_analysis_end``
     - Time of the last analysis cycle.
     - 2001-01-28T00:00:00Z
   * - ``cycle_period``
     - Interval in hours between analysis cycles.
     - 24
   * - ``time``
     - Time of the current analysis cycle.

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
   :widths: 20 55 25

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
   :widths: 20 45 35

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
   :widths: 20 45 35

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
     - 'qg'
   * - ``var_type``
     - Variable type.
     - 'field', or 'scalar'
   * - ``err_type``
     - Error distribution type.
     - 'normal'

The ``model_def`` entry is a dictionary, with model_name as keys pointing to a dictionary of model-specific configuration parameters.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Key
     - Description
     - Example
   * - ``config_file``
     - YAML configuration file for the model.

       Ideally, this is the only entry necessary and all

       the details can be defined in ``config_file``.

       However, additional entries added in ``model_def``
       
       will overwrite the settings from ``config_file``,

       making it easier to setup twin experiments.
     - 'models/qg/default.yml'
   * - ``model_env``
     - Initialization script for model.

       At runtime ``". {model_env}"`` will source
       
       this script before running the model forecast.
     - 'setup.src'
   * - ``model_code_dir``
     - Path to the model code.
     - '{nedas_root}/models/qg'
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
   * - ``ens_run_type``
     - Type of ensemble forecast to run.

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

       If ``use_synthetic_obs`` this is mandatory.
     - '{work_dir}/truth'

Observation definition
^^^^^^^^^^^^^^^^^^^^^^
The ``obs_def`` entry is a list, each item is a dictionary that defines one observation variable:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

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
     - 'qg'
   * - ``model_src``
     - Name of the model from which to compute the
     
       observation priors .
     - 'qg'
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
   * - ``impact_on_state``
     - List of impact factors of this observation

       on the state variables.

       The unlisted variable has a default impact of 1.0
     - { 'streamfunc': 0 },

       which turns off the

       impact on streamfunc

.. list-table:: Table 4. Breakdown of the observation error definition dictionary.
   :header-rows: 1
   :widths: 20 55 25

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
   :widths: 20 45 35

   * - Key
     - Description
     - Example
   * - ``config_file``
     - YAML configuration file for the dataset.

       Ideally, this is the only entry necessary and all

       the details can be defined in ``config_file``.

       However, additional entries added in ``dataset_def``
       
       will overwrite the settings from ``config_file``,

       making it easier to setup twin experiments.
     - 'dataset/qg/default.yml'
   * - ``dataset_dir``
     - Path to the dataset files
     - 'data'
   * - ``obs_window_min`` 
     - Start of the observation window, 

       hours relative to the analysis time.
     - -12
   * - ``obs_window_max``
     - End of the observation window,

       hours relative to the analysis time.
     - 12

Some additional parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Key
     - Description
     - Default
   * - ``use_synthetic_obs``
     - Whether to use synthetic observations generated

       from the truth.
     - True
   * - ``shuffle_obs``
     - Whether to randomize the order of observations.
     - False
   * - ``z_coords_from``
     - Where the reference vertical coordinates come from.
     - 'mean'

Perturbation
^^^^^^^^^^^^

The ``perturb`` entry is a list, each element is a dictionary with kwargs that will be passed to :func:`utils.random_perturb`
to perform the perturbation.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Key
     - Description
     - Default
   * - ``variable``
     - Name of variable to be perturbed.
     - 'streamfunc'
   * - ``model_src``
     - Name of the model the variable comes from.
     - 'qg'
   * - ``type``
     - Type of random perturbation.

       Use ',' to join multiple options
     - 'gaussian,exp'
   * - ``amp``
     - Amplitude of the perturbation.
     - 0.1
   * - ``hcorr``
     - Horizontal correlation length of the
     
       perturbation, in coordinate units
     - 15
   * - ``tcorr``
     - Temporal correlation length of the

       perturbation, in hours
     - 0
   * - ``bounds``
     - If set, the perturbed variable will

       remain in the value range.
     - [0, inf]

If no perturbation is needed, you can also leave ``perturb`` as None.

Assimilation method
^^^^^^^^^^^^^^^^^^^

The following parameters helps :func:`get_analysis_scheme` to locate the right subclass of :class:`AnalysisScheme`.
Currently only 'offline_filter' scheme is implemented.
For the specific ``filter_type``, the corresponding :class:`Assimilator` subclass will be chosen to perform the filter step.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Key
     - Description
     - Example
   * - ``analysis_type``
     - Type of analysis scheme to use. 
     - 'offline_filter'
   * - ``assim_mode``
     - Assimilation mode.
     - 'batch' or 'serial'
   * - ``filter_type``
     - Type of filter
     - 'ETKF' (for batch mode), 'EAKF' (for serial mode)

Filter-specific parameters are defined in ``filter`` dictionary.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Key
     - Description
     - Example
   * - ``config_file``
     - YAML configuration file for the filter algorithm.
     
       Ideally, this is the only entry necessary and all

       the details can be defined in ``config_file``.

       However, additional entries added in ``filter``
       
       will overwrite the settings from ``config_file``,

       making it easier to setup twin experiments.
     - 'assimilators/ETKF/default.yml'

Multiscale approach configuration:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Key
     - Description
     - Example
   * - ``nscale``
     - Number of scale components.
     - 1
   * - ``scale_id``
     - Current scale component index
     - 0
   * - ``decompose_obs``
     - Whether to decompose observations into scale

       components as well.
     - False
   * - ``resolution_level``
     - resolution level for the analysis grid 

       (0 is the analysis grid resolution,

       + reduces, - increases the resolution)
     - [0]
   * - ``character_length``
     - characteristic length (in grid coord unit)

       for each scale (large to small)
     - [16]
   * - ``localize_scale_fac``
     - scale factor for localization distances
     - [1]
   * - ``obs_err_scale_fac``
     - scale factor for observation error variances
     - [1]

Alignment technique configuration is stored in ``alignment`` as a dictionary:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Key
     - Description
     - Example
   * - ``interp_displace``
     - If True, use interpolation to find the variables

       on displaced analysis grid.

       If False, displace the grid coordinates directly.
     - False
   * - ``variable``
     - Name of the variable the alignment is based on.
     - 'streamfunc'
   * - ``method``
     - Optical flow method.
     - 'HS_pyramid'
   * - ``nlevel``
     - Number of resolution levels in pyramic approach
     - 5
   * - ``smoothness_weight``
     - Weight in cost function to enforce

       smoothness of displace vector field
     - 1

Covariance inflation parameters are stored in the ``inflation`` entry as a dictionary.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Key
     - Description
     - Example
   * - ``type``
     - Type of inflation (posterior/prior, multiplicative/RTPP).
     - 'posterior,RTPP'
   * - ``adaptive``
     - Whether to run an adaptive inflation scheme.
     - True 
   * - ``coef``
     - Static inflation coefficient.
     - 1.0

Covariance localization parameters are stored in the ``localization`` entry as a dictionary.

.. list-table::
   :header-rows: 1
   :widths: 20 65 15

   * - Key
     - Description
     - Example
   * - ``horizontal``
     - Type of horizontal localization function. Distance-based (GC, step, exp)
     
       or correlation-based (NICE)
     - 'GC'
   * - ``vertical``
     - Type of vertical localization function.
     - 'GC'
   * - ``temporal``
     - Type of temporal localization function.
     - 'exp'

Diagnostic methods
^^^^^^^^^^^^^^^^^^
The ``diag`` entry is a list, each element is a dictionary defining a diagnostic method to be run.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Key
     - Description
     - Example
   * - ``method``
     - Name of the diagnostic method
     - 'misc.convert_output'
   * - ``config_file``
     - YAML configuration file for the method.

       Ideally, this is the only entry necessary

       with all details in ``config_file``.

       However, additional entries added below

       will overwrite the settings.
     - 'diag/misc/convert_output/default.yml'