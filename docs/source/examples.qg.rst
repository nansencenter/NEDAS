QG model
========

A quasi-geostrophic model (qg), written in Fortran by `Dr. Shafer Smith <https://cims.nyu.edu/~shafer/tools/index.html>`_,
is implemented in NEDAS as a test model.

A Docker image provides a demonstration of the offline filter analysis scheme.

.. code-block:: bash

   docker pull myying/nedas-qgmodel-benchmark

Start a container with

.. code-block:: bash

   docker run -it --rm -v YOUR_WORK_PATH:/work myying/nedas-qgmodel-benchmark

where ``YOUR_WORK_PATH`` is the local disk location where you want to save the output files.

When the container starts running, you can first run 

.. code-block:: bash

   ./prepare_files.sh

to generate the truth and intial ensemble member files.

Then start the cycling data assimilation:

.. code-block:: bash

   python -m NEDAS -c /app/config.yml

Files generated at runtime is located in ``/work/cycle``,
additional netCDF output files are saved at ``/work/output``.

For benchmarking of the performance, you can change parameters by adding runtime arguments:

- ``--nens=NENS`` changes the ensemble size (int)
- ``--nproc=NPROC`` changes the number of processors to use (int)

To make more detailed changes (such as localization and inflation parameters),
you can make a copy of the YAML configuration file

.. code-block:: bash
   cp /app/config.yml /work/new-config.yml

then edit it externally from ``YOUR_WORK_PATH/new-config.yml`` and run it with

.. code-block:: bash
   python -m NEDAS -c /work/new-config.yml

You can also turn on debug mode ``--debug=on`` to have more detailed runtime messages.
