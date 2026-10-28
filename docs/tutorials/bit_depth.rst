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

   lut.raw.dtype      # numpy.uint8  -- actually stored
   lut.nbytes         # 107,811 bytes (a float 33^3 is 431,244)
   lut.volume         # float32 [0, 1] -- always, for the kernels

Quantising at 8 bits divides the stored size by **4**; at 16 bits, by **2**. The
public ``lut.volume`` stays float32 in ``[0, 1]`` for the kernels; only the stored
buffer is compact, and :meth:`lutlab.LUT.nbytes` / ``lut.cost().nbytes`` report
the real size. Above 16 bits there is no smaller integer buffer: the values are
quantised to the requested level count but stored as float32, so 32 bits is the
cap and float32 is its precision.

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
   * - ``17 .. 32``
     - ``float32``
     - 431,244
     - no smaller buffer

Supported values are ``1 .. 32``. Depths 1..8 use ``uint8``, 9..16 use
``uint16``, and 17..32 use ``float32``.

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

* **``.json``** is lossless: it stores the compact grid *and* ``bit_depth``, so
  the tag survives a round-trip at every depth.
* **Hald ``.png``** uses an 8-bit container for ``bit_depth <= 8`` and a 16-bit
  container above that. The PNG itself only says which container was used, so a
  10/12/17..32-bit LUT reads back as 16-bit and a 1..7-bit LUT as 8-bit; the
  values stay on the same levels. Custom domains and non-uniform grids cannot
  be stored in a Hald pattern and are rejected by ``write``.
* **``.cube`` / ``.npy``** store values but not the tag. ``.npy`` writes the
  compact buffer only for 8/16-bit LUTs and float32 otherwise, so the tag is
  lost; use ``.json`` when it matters. ``.cube`` is decimal text with no tag.

Throughput
----------

For the LUT sizes used in practice (``N <= 65``), quantisation does **not**
change apply throughput: the kernels interpolate on a float volume either way and
the image, not the LUT, dominates. The win is **memory**, not speed. A speed win
would need a LUT larger than the L2 cache *and* a kernel that interpolates
directly on the compact dtype -- that is a separate kernel change, not something
``bit_depth`` does today.
