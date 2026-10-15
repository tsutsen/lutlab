lutlab.apply
============

One ``apply()`` over every backend (numpy / numba / gpu / fused) and every
interpolation scheme -- same math, so the numbers are comparable.
``segmented_apply`` is the Art 5 version that blends per-region LUTs with
soft masks.

.. autosummary::

   lutlab.apply.apply
   lutlab.apply.segmented_apply
   lutlab.apply.backends

.. automodule:: lutlab.apply
   :members:
