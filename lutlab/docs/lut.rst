lutlab.lut
==========

``LUT`` is a colour lookup table: a regular grid of colours with a single
apply path. This is the type you hold a LUT as; everything else in lutlab
accepts a ``LUT`` (or a bare volume array) and round-trips through its
canonical layout.

A ``LUT`` wraps a ``vol`` of shape ``(D,)*n + (3,)``: the first ``n`` axes are
the *input* dimensions (axis 0 = input dimension 0, i.e. R for a 3D LUT) and
the last axis is the 3 output RGB channels. The number of input axes sets the
dimensionality -- a 2D LUT is ``(D, D, 3)``, a 3D LUT is ``(D, D, D, 3)``, and
a 4D LUT is ``(D, D, D, D, 3)`` (for example three channels plus a luminance
input).

Attributes
----------

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Name
     - Type
     - Meaning
   * - ``vol``
     - ``np.ndarray``
     - The raw grid, shape ``(D,)*n + (3,)``, float32.
   * - ``name``
     - ``str``
     - Human label for the LUT.
   * - ``D``
     - ``int``
     - Grid size along every input axis.
   * - ``ndim``
     - ``int``
     - Number of input dimensions (2, 3 or 4).
   * - ``shape``
     - ``tuple``
     - ``vol.shape`` (read-only property).

Methods
-------

.. autosummary::

   lutlab.lut.LUT.apply
   lutlab.lut.LUT.sample
   lutlab.lut.LUT.resample
   lutlab.lut.LUT.to_cube
   lutlab.lut.LUT.from_cube
   lutlab.lut.LUT.to_json
   lutlab.lut.LUT.from_json
   lutlab.lut.LUT.from_numpy
   lutlab.lut.LUT.storage_bytes
   lutlab.lut.LUT.delta_from_identity
   lutlab.lut.LUT.tone_curves
   lutlab.lut.LUT.max_delta

Constructors
------------

.. autosummary::

   lutlab.lut.identity

.. automodule:: lutlab.lut
   :members:
