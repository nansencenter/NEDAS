Vort2D model
============

A step-by-step tutorial demonstrating data assimilation with NEDAS using the 2D vorticity (``vort2d``) model.
The notebook walks through a complete OSSE (Observing System Simulation Experiment):
generating a model truth, creating a perturbed ensemble, assimilating synthetic observations,
and visualising error distributions and increments interactively.

**Topics covered:**

- Configuring an experiment with a YAML file and the ``Config`` object
- Generating a verifying truth and synthetic observations
- Building a perturbed ensemble (vortex position and background flow)
- Running one assimilation cycle: observation priors, increments, state update
- Assimilating a single wind observation to inspect ensemble error geometry
- Interactive widgets and animations for exploring observation–state correlations

The notebook can be run in several environments:

- `Google Colab <https://colab.research.google.com/github/myying/NEDAS_tutorials/blob/main/1.step_by_step_with_vort2d_case.ipynb>`_
- Docker (see below)
- Native Python — refer to the `environment setup guide <https://github.com/myying/NEDAS_tutorials/blob/main/python_env_setup.md>`_

.. code-block:: bash

   docker pull myying/nedas-tutorials
   docker run -it --rm -p 8888:8888 myying/nedas-tutorials

Then open the URL printed in the terminal and navigate to
``1.step_by_step_with_vort2d_case.ipynb``.

The full notebook is available on
`GitHub <https://github.com/myying/NEDAS_tutorials/blob/main/1.step_by_step_with_vort2d_case.ipynb>`_.
