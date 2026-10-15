lutlab.extract
==============

LUT extraction: read a LUT back from a (misaligned) Hald render, or from a
before/after image pair. The two entry points are ``hald_extract`` and
``extract``; the rest of the names below are the low-level routines they
build on.

.. autosummary::

   lutlab.extract.hald_extract
   lutlab.extract.extract

**Re-exported from ``robust_hald``**

.. autosummary::

   lutlab.extract.naive_extract
   lutlab.extract.farneback_flow
   lutlab.extract.flow_aligned_extract
   lutlab.extract.grid_aligned_extract
   lutlab.extract.hald_percell_extract
   lutlab.extract.structure_patches
   lutlab.extract.fuzzy_correspond
   lutlab.extract.cluster_extract
   lutlab.extract.barycentric_extract
   lutlab.extract.refine_lut
   lutlab.extract.grid_flow

.. automodule:: lutlab.extract
   :members:
   :imported-members:
