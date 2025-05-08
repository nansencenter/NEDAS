QG model
========

A quasi-geostrophic model (qg) in Fortran


.. code-block:: bash

   docker pull myying/nedas-qgmodel-benchmark

Start a container with

.. code-block:: bash

   docker run -it --rm myying/nedas-qgmodel-benchmark

Once inside the container, run ``./prepare_files.sh`` to generate the truth and initial ensemble files, then run ``./run_expt.sh`` to start the offline filter analysis scheme.

Results will be saved in ``/work``, but if you exit the container it will be lost.
To keep a copy on your local disk, add ``-v YOUR_WORK_PATH:/work`` in the ``docker run`` command options when starting the container. Then the ``/work`` directory will become a shared volume with ``YOUR_WORK_PATH``.
