Vort3D model
============

A minimal 3D tropical cyclone model following `Zhu, Smith & Ulrich (2001) <https://doi.org/10.1175/1520-0469(2001)058%3C1801:ANMFTC%3E2.0.CO;2>`_:
sigma-coordinate primitive equations on an f/beta-plane, with ``nz`` free-atmosphere
layers plus one boundary layer at the bottom, surface fluxes, radiative cooling,
explicit condensation, and a convective closure (the paper's own ``ooyama`` scheme,
or a simplified ``betts_miller`` column relaxation for ``nz`` other than 2).

The prognostic state is wind (:math:`u,v`), potential temperature (:math:`\theta`),
and moisture (:math:`q`) on all ``nz + 1`` layers, plus a single-level surface
pressure field (:math:`p^*`, column mass). Vortex position, intensity, and size,
the background steering flow, and the reference Coriolis parameter ``f0`` are all
configurable, making it a lightweight testbed for ensemble DA experiments involving
vortex tracking and position error.

See :doc:`NEDAS.models.vort3d` for the model interface and
:doc:`NEDAS.datasets.vort3d` for the corresponding synthetic observation dataset.
Default parameters are documented in ``NEDAS/models/vort3d/default.yml``.

.. note::
   No step-by-step tutorial notebook is available yet for this model
   (unlike ``vort2d``, ``qg``, and ``nextsimdg`` above).
