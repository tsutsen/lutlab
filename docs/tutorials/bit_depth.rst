Bit depth
=========

.. _doc-bitdepth:

.. _sec-bit-depth:

A :class:`lutlab.LUT` normally stores its values as float. Giving it a
``bit_depth`` instead **quantises the volume once at construction and stores the
result compactly** -- this is a real lever on stored size and memory, and it
changes what a Hald ``.png`` round-trip does.

.. code-block:: python

   import numpy as np, lutlab

   lut = lutlab.LUT(np.random.rand(33, 33, 33, 3), bit_depth=8)

   lut.dtype          # numpy.uint8  -- actually stored
   lut.nbytes         # 107,811 bytes (a float 33^3 is 431,244)
   lut.volume         # float32 [0, 1] -- always, for the kernels

Quantising at 8 bits divides the stored size by **4**; at 16 bits, by **2**. The
public ``lut.volume`` stays float32 in ``[0, 1]`` for the kernels; only the stored
buffer is compact, and :meth:`lutlab.LUT.nbytes` / ``lut.cost().nbytes`` report
the real size.

.. list-table::
   :header-rows: 1

   * - ``bit_depth``
     - Stored as
     - Bytes (33^3)
     - Versus float
   * - ``None`` (default)
     - ``float32``
     - 431,244
     - baseline
   * - ``16``
     - ``uint16``
     - 215,622
     - **2×** smaller
   * - ``8``
     - ``uint8``
     - 107,811
     - **4×** smaller

Supported values are ``1, 2, 4, 8, 16``.

Changing depth
--------------

:meth:`lutlab.LUT.to_bit_depth` moves a LUT to a new depth:

.. code-block:: python

   lut8   = lut.to_bit_depth(8)
   lut16  = lut8.to_bit_depth(16)   # warns: adds no precision (up-conversion)
   lut4   = lut.to_bit_depth(4)     # silent: a normal lossy down-conversion

.. note::

   *Down*-conversion is a normal lossy quantise and is silent. *Up*-conversion
   (e.g. 8-bit to 16-bit) cannot recover information, so it warns.

How it interacts with formats
-----------------------------

* **Hald ``.png``** is the one format that round-trips the tag: a quantised LUT
  is written at its depth (8-bit PNG for ``bit_depth <= 8``, 16-bit PNG for
  ``> 8``) and read back at the same depth, *losslessly*.
* **``.cube`` / ``.json`` / ``.npy``** store the float upcast and drop the tag
  on reload -- a quantised LUT reloaded from them comes back float.

Throughput
----------

At the LUT sizes used in the benchmark (``N <= 65``), quantisation does **not**
change apply throughput: the kernels interpolate on a float volume either way and
the image, not the LUT, dominates. The win is **memory**, not speed. A speed win
would need a LUT larger than the L2 cache *and* a kernel that interpolates
directly on the compact dtype -- that is a separate kernel change, not something
``bit_depth`` does today.
