# Developer's Guide

To use NEDAS for your own models/observations, please read the detailed documentation for `models` and `dataset` modules, and create a module with functions to interface with the new models and/or dataset files. In the workflow chart the user-provided functions are highlighted in orange.

If you are considering DA experiments for a model, typically some Python diagnostic tools for the model state variables already exist, so the work for implementing the modules shall not be too heavy. Essentially you need to provide functions such as `read_var` to receive some key word arguments (variable name, time, member, vertical index, etc.) and return a 2D field containing the corresponding model state variable.

For observations, we expect you to already have some preprocessing scripts to read the raw dataset, quality control and screen for valid observations for the analysis domain, etc. These can be implemented in the `read_obs` function. Some steps in preprocessing are more involved: super-observation, uncertainty estimation, and extraction of information matching the model-resolved scales. We suggest you consult DA experts to implement these steps.

List of currently supported models and observations:

* The TOPAZ system coupled ocean (HYCOM) and sea ice (CICE4) model, with satellite obserations and insitu profiler data.

* The next-generation sea ice model (neXtSIM), with SAR-image-based sea ice drift and deformation observations.

and planned developement for:

* The Weather Research and Forecast (WRF) model (Polar WRF), with satellite observations.

* ECOSMO biogeochemistry model, with ocean color data.
