Interpolation schemes
=====================

.. _sec-schemes:

A pixel's input colour generally falls *between* the LUT's grid nodes. The
**scheme** decides how to combine the surrounding nodes into a single output
colour. All three are implemented identically on every backend, so a scheme is
comparable across CPUs and GPUs.

Nearest
-------

.. _scheme-nearest:

Snap the input to the single nearest node. Fastest, and it is the only scheme
that preserves the LUT's exact node values -- but it introduces visible banding
and a bias toward black (it floors, it does not round to the nearest value).
Use it for debugging, not for output.

.. code-block:: python

   out = lut(img, interp="nearest")

Trilinear
---------

.. _scheme-trilinear:

Blend the eight nodes of the cell the input falls in, weighting each corner by
its distance. This is the standard 3D-LUT interpolation and the default. It is
smooth and cheap, and it is the scheme every external baseline (Pillow, ffmpeg
``lut3d``, colour-science, OpenColorIO) uses, so it is the one to quote in
comparisons.

.. code-block:: python

   out = lut(img, interp="trilinear")

Tetrahedral
-----------

.. _scheme-tetrahedral:

Split the cell into six tetrahedra and blend only the four nodes of the
tetrahedron the input falls in, instead of all eight corners. On CPU it is
essentially the same speed as trilinear; it is included for completeness and to
confirm the two agree.

.. code-block:: python

   out = lut(img, interp="tetrahedral")

Which one to use
----------------

.. list-table::
   :header-rows: 1

   * - Scheme
     - Reads/px
     - Typical use
     - Notes
   * - ``nearest``
     - 1
     - debug, banding check
     - fastest
   * - ``trilinear``
     - 8
     - **default**, comparisons
     - matches every baseline
   * - ``tetrahedral``
     - 4
     - parity check
     - ~= trilinear on CPU

The :ref:`fused <backend-fused>` GPU backend supports trilinear only; the
``numpy``, ``numba`` and ``torch`` backends support all three.
