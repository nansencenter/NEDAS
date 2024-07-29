# Model source modules

For each model component, a `Model` class should provide the interface to the model state variables in the restart files, it should also provide methods to initiate and run the model forecasts.

The model class methods will be called by `assim_tools` with `**kwargs` to obtain information on a 2D field as part of the model state. In runtime the kwargs are obtained from `state_info` records parsed by `assim_tools.state`.

List of common inputs and keys in `**kwargs`:

* `path` is the directory in which the module functions will search for model restart files. In `assim_tools` the path will be generated based on environment variables from `config`.

* `name` = name of the state variable, defined in `self.variables` keys.

* `dtype` = data type for the variable ('double', 'float' or 'int')

* `is_vector` = True if the state variable is a vector field, or False if a scalar field.

* `units` = physical units for the state variable.

* `time` = `datetime` object for the time of interest.

* `member` = id for the ensemble member (0, 1, ..., nens-1).

* `k` = id for the vertical level, full list defined in `variables[name]['levels']`.

List of necessary properties and methods in `Model` class:

* `self.grid` is the Grid object describing the model 2D domain.

* `self.variables` is a dictionary with keys `name` (variable names) each pointing to a record dictionary with keys:

	'name' : native model variable name, if variable is vector field a tuple of u- and v-component names is provided here.

	'dtype' : data type of the variable,

	'is\_vector' : True if variable is a vector field,

	'restart\_dt' : model restart interval in hours,

	'levels' : vertical level indices for variable, and

	'units' : physical units for variable.

* `filename(self, **kwargs)` returns the name of a restart file matching the keys in kwargs.

* `read_grid(self, **kwargs)` sets the Grid object `self.grid` describing the model 2D domain. See Grid class documentation for more details how to create an instance.

* `write_grid(self, **kwargs)` will output information in `self.grid` to model restart files. This function is only needed if model grid is updated by the assimilation scheme, if your model has a fixed grid, this function can be just a place holder with `pass`.

* `read_mask(self, **kwargs)` (optional) returns the array with same size of model grid, True if the grid point is masked. This mask is useful to reduce memory use if the model domain contains area (such as land for ocean models) where state variables are not defined.

* `read_var(self, **kwargs)` returns the variable 2D field `fld`, which is an array of shape `self.grid.x.shape` if scalar field or `(2,)+self.grid.x.shape` if vector field.

	This function should find the restart file using `filename()`, read the native variable from the file using information defined in `variables`, convert units using `conversion.units_convert()`, and perform other operation specific to the model (you can define in the function).

	Non-prognostic variables can also be defined in `variables`, in `read_var()` a procedure of obtaining those variables can be defined to return it. The variable can be a diagnostic variable, an observed variable, or just a model parameter. If you provide a procedure to compute an observed variable in `read_var()`, this procedure will be used in `assim_tools.obs.state_to_obs()` to compute observation priors when assimilating this type of observation.

* `write_var(self, fld, **kwargs)` outputs the prognostic variable 2D field `fld` back to the model restart file. It should perform the reverse process in `read_var()` (unit conversion and other things you defined).

	For now, we assume that the restart file already exists (for assimilating purposes this is true), so we are opening the restart files in 'r+' mode and overwrite the existing variables with new values.

* `self.z_units` is the default physical units for the model z coordinates.

* `z_coords(self, **kwargs)` returns a 2D field, similar to the field `fld` returned by `read_var()` but replaced by the z coordinates at each grid point of that field. If 'z\_units' is provided in kwargs, the function should convert z from `self.z_units` to the desired z\_units (useful when observation is defined with different z units), otherwise the default `self.z_units` will be used.

* `generate_initial_condition(self, task_id, task_nproc, **kwargs)` links and prepares the initial condition files at the first analysis cycle of an experiment. `task_id` is given at runtime to parallel execute the function; `task_nproc` is the number of processors used for each task. In addition to the `kwargs` given above, `time_start`, `time_end` and `cycle_period` will also be provided to define the experiment time period.

* `run(self, task_id, task_nproc, **kwargs)` runs the model forecast for each cycle. `task_id` and `task_nproc` are also for parallel execution of the run code. In `kwargs`, `time` should be provided for the current cycle datetime object, `forecast_period` is the forecast duration in hours, `output_dir` (optional) is additional output directory to make a copy of the resulting restart files (for analysis at next cycle).


