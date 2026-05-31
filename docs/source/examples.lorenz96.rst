Lorenz96 model
==============

A tutorial that validates the NEDAS ensemble Kalman filter implementation against
the classic `Lorenz 1996 <https://doi.org/10.1142/9789812833273_0005>`_ model.

The model is a one-dimensional, cyclic 40-variable system representing mid-latitude
zonal atmospheric circulation. With a forcing parameter of F = 8 the system is chaotic,
making it a standard benchmark for data assimilation algorithms. One model time step
(Δt = 0.05) corresponds to approximately 6 hours.

Results are cross-checked against the deterministic EnKF derivation in Bocquet's
lecture notes (section 5.5).

**Topics covered:**

- Configuring ensemble size, observation error variance, localization radius, and observation network
- Running the main cycling scheme (with Numba JIT compilation on first cycle)
- Collecting in-memory diagnostics over multiple assimilation cycles
- Animating ensemble spread and truth trajectory
- Plotting error time series to verify filter convergence

The notebook can be run in several environments:

- `Google Colab <https://colab.research.google.com/github/myying/NEDAS_tutorials/blob/main/2.validation_with_lorenz96.ipynb>`_
- Docker (see below)
- Native Python — refer to the `environment setup guide <https://github.com/myying/NEDAS_tutorials/blob/main/python_env_setup.md>`_

.. code-block:: bash

   docker pull myying/nedas-tutorials
   docker run -it --rm -p 8888:8888 myying/nedas-tutorials

Then open the URL printed in the terminal and navigate to
``2.validation_with_lorenz96.ipynb``.

The full notebook is available on
`GitHub <https://github.com/myying/NEDAS_tutorials/blob/main/2.validation_with_lorenz96.ipynb>`_.
