QG model
========

A quasi-geostrophic model (qg), written in Fortran by `Dr. Shafer Smith <https://cims.nyu.edu/~shafer/tools/index.html>`_,
is implemented in NEDAS as a test model.

The model describes the evolution of the streamfunction :math:`\psi` in a two-layer,
doubly periodic domain under quasi-geostrophic dynamics (kmax = 127, 256 × 256 grid).
Its realistic two-dimensional spatial structure makes it a standard benchmark for
localization-based ensemble DA algorithms.

**Topics covered in the tutorial notebook:**

- Configuring ensemble size, observation network, localization radius, and inflation
- Running multi-cycle OSSE experiments with the Fortran QG model
- Reading and plotting RMSE versus DA cycle to verify filter convergence
- Visualizing streamfunction fields (truth, prior mean, posterior mean)
- Comparing batch (ETKF) and serial (EAKF) ensemble Kalman filter strategies
- Correcting position errors with the multiscale alignment updator (Horn-Schunck optical flow)

The notebook can be run in several environments:

- Docker (see below)
- Native Python — refer to the `environment setup guide <https://github.com/myying/NEDAS_tutorials/blob/main/python_env_setup.md>`_

.. code-block:: bash

   docker pull myying/nedas-tutorials
   docker run -it --rm -p 8888:8888 myying/nedas-tutorials

Then open the URL printed in the terminal and navigate to
``3.multiscale_alignment_with_qgmodel.ipynb``.

The full notebook is available on
`GitHub <https://github.com/myying/NEDAS_tutorials/blob/main/3.multiscale_alignment_with_qgmodel.ipynb>`_.
