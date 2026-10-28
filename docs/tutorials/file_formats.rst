File formats
============

.. _doc-formats:

:ref:`LUT.load(path) <class-LUT-load>` and :ref:`lut.save(path) <class-LUT-save>`
dispatch on the file extension. The
same extension is how a LUT is told which format to write.

.. list-table::
   :header-rows: 1

   * - Ext.
     - What it is
     - Notes
   * - ``.cube``
     - Standard 3D LUT (DaVinci Resolve)
     - written with the standard header (``TITLE``, ``LUT_3D_SIZE``, ...); ``DOMAIN_MIN`` / ``DOMAIN_MAX`` honoured
   * - ``.json``
     - Structured JSON
     - lossless: values, axes, domain, space and the bit-depth tag
   * - ``.npy``
     - Raw ``numpy`` volume
     - values only, no metadata (the bit-depth tag is lost)
   * - ``.png``
     - A Hald pattern (see below)
     - quantised to an 8- or 16-bit PNG; see the Hald notes below

.. code-block:: python

   lut.save("grade.cube")     # write as a .cube
   lut2 = lutlab.LUT.load("grade.cube")

The ``.cube`` writer emits the standard header and honours ``DOMAIN_MIN`` /
``DOMAIN_MAX``; it also still reads the pre-0.2 header. ``.json`` is lossless
and keeps the :ref:`bit-depth tag <doc-bitdepth>`. ``.npy`` stores the compact
buffer only for 8/16-bit LUTs and float32 for every other depth, so its tag is
lost on read; use ``.json`` when the tag matters. ``.cube`` is decimal text and
carries no tag either.

Hald as a format
----------------

A **Hald pattern** is the standard image for encoding a 3-D LUT in a single
picture, so that the picture is the LUT laid out as a texture. Two pixel
layouts exist and ``lutlab.formats.hald`` reads and writes both:

``layout="canonical"`` (the standard)
   The layout ``ImageMagick``'s ``-hald-clut`` uses and the classic Hald
   specification describes. Level ``N`` gives an image of ``N**3`` pixels per
   side (``hald:4`` -> 64x64, ``hald:8`` -> 512x512, ``hald:16`` ->
   4096x4096); the table carries ``N**2`` values per axis (so ``hald:8`` is a
   64x64x64 table) and every node is exactly one pixel. The pixels are a
   flattened 3-D buffer -- width is red, height is green, depth is blue, with
   blue outermost -- so the pixel at ``(x, y)`` holds the node
   ``r = x % C``, ``g = (y % N) * N + x // C``, ``b = y // N`` with
   ``C = N**2``. Files written this way are interchangeable with ImageMagick.

``layout="lutlab"`` (the default)
   A lutlab variant: the ``L**3`` nodes are laid out on a
   ``ceil(sqrt(L**3)) x ceil(sqrt(L**3))`` grid of cells, each node occupying a
   ``cell x cell`` pixel block, so the image side is
   ``ceil(sqrt(L**3)) * cell``. This gives two independent "sizes" -- the table
   density (``levels``) and the cell size (``cell``); a larger cell survives
   lossy compression better, and does not change the accuracy of a PNG
   round-trip.

* **Write** renders the LUT onto the requested pattern and saves a PNG. The PNG
  is 8-bit for ``bit_depth <= 8`` and 16-bit above, chosen from the LUT's
  :ref:`bit depth <doc-bitdepth>`. A custom domain or a non-uniform grid cannot
  be represented by the unit-cube pattern and raises -- use ``.json`` for those.
* **Read** parses the pattern back into a :class:`lutlab.LUT`, inferring only
  the container depth (8 or 16) from the file. A 10/12/17..32-bit LUT therefore
  reads back as 16-bit and a 1..7-bit LUT as 8-bit; the values stay on the same
  levels. Pass ``bit_depth=`` to relabel the result.

.. code-block:: python

   lut.save("grade.png")                       # lutlab variant; 8-bit for a float LUT
   lut2 = lutlab.LUT.load("grade.png")
   lut2.allclose(lut)                          # exact only if lut was already 8-bit

   lut.save("hald8.png", layout="canonical", level=8)   # ImageMagick-compatible
   lut3 = lutlab.LUT.load("hald8.png", layout="canonical")

.. note::

   The two layouts can produce images of the same size, so the reader does not
   guess: pass ``layout=`` explicitly, and for the lutlab variant also
   ``levels=`` / ``cell=``.

.. note::

   Writing a Hald chart needs OpenCV (``cv2``) and a cubic (3-D) LUT. Reading
   is the same. Install it with ``pip install "lutlab[png]"`` (or
   ``opencv-python-headless``). The pattern itself is pure numpy; only the PNG
   I/O needs ``cv2``, because PIL and imageio in current releases cannot write
   16-bit *RGB* PNG.
