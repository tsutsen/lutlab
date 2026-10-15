lutlab.formats
==============

LUT format conversion. ``.cube`` is the common 3D interchange format (and the
only one the GPU fast path can use); the compact JSON round-trip is lossless
and handles 2D/4D too. ``resample`` changes grid resolution; ``validate``
cross-checks the backends.

.. autosummary::

   lutlab.formats.to_cube
   lutlab.formats.from_cube
   lutlab.formats.to_json
   lutlab.formats.from_json
   lutlab.formats.resample
   lutlab.formats.validate

.. automodule:: lutlab.formats
   :members:
