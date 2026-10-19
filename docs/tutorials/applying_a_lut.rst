Applying a LUT
==============

.. _doc-applying:

Applying a LUT to an image is one call. Everything else -- the scheme, the
backend, the device -- is an argument to that one call.

.. code-block:: python

   import numpy as np
   import lutlab

   lut = lutlab.testing.random_lut(17, seed=0)
   img = np.random.rand(720, 1280, 3).astype(np.float32)

   out = lut(img)                                   # defaults: trilinear, auto
   out = lut(img, interp="tetrahedral")             # pick the scheme
   out = lut(img, interp="trilinear", backend="numba")  # pin the backend

The ``LUT`` is callable -- ``lut(img, ...)``. The same lookup for a bare
(transform, array) pair is :ref:`lutlab.kernels.apply <func-kernels-apply>`. The
result has the same shape and type as ``img``.

Arguments
---------

.. list-table::
   :header-rows: 1

   * - Argument
     - Meaning
     - Default
   * - ``interp``
     - Interpolation :ref:`scheme <sec-schemes>`
     - ``"trilinear"``
   * - ``backend``
     - Which :ref:`backend <doc-backends>` to use
     - ``"auto"``
   * - ``device``
     - ``"cpu"`` / ``"cuda"`` / ``None``
     - ``None`` (where data is)
   * - ``context``
     - Extra input for a 4D LUT (or a per-pixel mask)
     - ``None``
   * - ``check_finite``
     - Validate that inputs are finite
     - ``None``

The rules that apply to every backend: inputs are float in the LUT's ``domain``
(default ``[0, 1]``), out-of-range values are clamped, NaN/inf raise, and
integer images raise (normalise them to float in [0, 1] first). See
:ref:`Core concepts <doc-design>` for the full contract.

Resampling a LUT
----------------

You can move a LUT onto a different grid without touching the images, e.g. to
downsize a 33^3 LUT for a fast preview:

.. code-block:: python

   lut_small = lut.resample(17)     # resample onto a uniform 17-per-axis grid

This multi-linearly interpolates the LUT itself and returns a new
:ref:`LUT <class-LUT>`. A quantised LUT keeps its bit depth through resampling
(see :ref:`Bit depth <doc-bitdepth>` and
:ref:`LUT.resample <class-LUT-resample>`).
