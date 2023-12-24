# Dataset source modules

A dataset source module provides a set of standard functions called by `assim_tools` with `**kwargs` to obtain information on an observation sequence. In runtime the kwargs are obtained from `obs_info` records parsed by `assim_tools.obs`.


List of common inputs:

* `path` is the directory in which the module functions will search for observation data files. In `assim_tools` the path will be generated based on environment variables from `config`.

* `grid` is the Grid object for the model 2D domain, which helps screening for observations inside the domain.

* `mask` is the mask for grid points where state variables are not defined, obtained from model module `read_mask()`, which helps screening for observations located near valid model grid points.

* `z` is the z coordinates for model vertical levels, of shape `(nz,)+grid.x.shape`. The z coordinates help finding the observations close to 

List of keys in `**kwargs`:

* `name` = name of the observation, defined in `variables` keys.

* `dtype` = data type for the observation ('double', 'float', or 'int').

* `is_vector` = `True` if the observation is a vector, or `False` if a scalar.

* `units` = physical units for the observation.

* `time` = `datetime` object for the observation time, not the exact measurement time for individual data points, but rather the index time for the analysis cycle.

* `obs_window_min`, `obs_window_max` (in hours) defines the observation window, observations valid from `time`+`obs_window_min` to `time`+`obs_window_max` will be considered for assimilation for analysis cycle at `time`.

* `z_units` = physical units of vertical coordinate for the observation


List of necessary functions/parameters:

* `variables` is a dictionary with keys `name` (observation names) each pointing to a record dictionary with keys:

	'dtype' : data type of the observed variables,

    'is\_vector': True if observed variable is a vector,

    'z\_units', units for the vertical coordinates of observed variables, and

    'units' : physical units for observed variables.


* `filename(path, **kwargs)` returns the name of an observation dataset file matching the keys in kwargs.


* `read_obs(path, grid, mask, z, **kwargs)` returns the observation sequence `obs_seq` given the keys in kwargs. The `obs_seq` is a dictionary with keys: 

	'obs' : a list of observed values (measurements), one measurement is either a number (scalar) or an array of [2] (vector).

	't' : a list of `datetime` objects for the observation time of each measurement.

	'z' : a list of vertical z coordinates for each measurement.

	'y' : a list of y coordinates for each measurement.

	'x' : a list of x coordinates for each measurement.


* `random_network(**kwargs)` generates a random realization of observing network for observation type `name`, it is used in generation of synthetic observations from model nature runs (truth). The function returns `obs_seq` but only the coordinates 't', 'z', 'y', 'x' are needed. The observed values will be generated later by `state_to_obs()` instead for the synthetic observations.


* `obs_operator` is a dictionary with keys `name` each pointing to a dictionary with keys `model` each pointing to a function `get_<name>_from_<model>` defined in the module. The function describes how to compute observation type `name` for model `model` restart files, and it returns the desired `obs_seq`. These observation operators also provide one of means in `state_to_obs()` to compute observation priors during assimilation.

