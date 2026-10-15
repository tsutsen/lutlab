.. lutlab documentation master file

lutlab
======

A small, self-contained library for the LUT math that appears in every paper in
this repo -- extraction, the typed ``LUT`` type, unified application across
backends, format conversion, and the metrics the results are scored with. One
implementation per operation, so the numbers are directly comparable.

**Try it**
----------

.. code-block:: python

   import numpy as np, lutlab

   img = np.random.rand(8, 8, 3).astype(np.float32)
   lut = lutlab.identity(17)          # the 17^3 identity LUT
   out = lutlab.apply(lut, img)       # trilinear, GPU if a device is present
   print(lutlab.metrics.e00(out, img))  # ~0

Run the checks: ``python -m lutlab selftest`` · run the benchmark:
``python -m lutlab bench``.

**Contents**
------------

.. toctree::
   :maxdepth: 1
   :caption: Modules
   :name: modules

   interp
   lut
   apply
   formats
   metrics
   hald
   extract
   segmented
   bench

**Shape convention**
--------------------

A volume is ``(D,)*n + (3,)`` -- the last axis is output RGB, the first ``n``
are input dimensions (``n=2,3,4`` for 2D/3D/4D LUTs). A ``lutlab.LUT`` is the
same volume plus a name and methods; a bare array is accepted everywhere too.
