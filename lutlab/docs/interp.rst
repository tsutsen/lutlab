lutlab.interp
=============

Interpolation: which corners of the LUT grid to blend for a point that falls
*between* grid cells. ``interp`` is the entry point and works on 2D, 3D and
4D LUTs; the other names are the individual kernels it dispatches to.

.. autosummary::

   lutlab.interp.interp
   lutlab.interp.interp_nearest
   lutlab.interp.interp_multilinear
   lutlab.interp.interp_tetrahedral
   lutlab.interp.interp_name

.. automodule:: lutlab.interp
   :members:
