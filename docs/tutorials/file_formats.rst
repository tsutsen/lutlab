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
     - the format ``lutlab`` itself round-trips in the test suite
   * - ``.npy``
     - Raw ``numpy`` volume
     - lossless, no metadata
   * - ``.png``
     - A Hald pattern (see below)
     - quantised to the LUT's :ref:`bit depth <doc-bitdepth>`

.. code-block:: python

   lut.save("grade.cube")     # write as a .cube
   lut2 = lutlab.LUT.load("grade.cube")

The ``.cube`` writer emits the standard header and honours ``DOMAIN_MIN`` /
``DOMAIN_MAX``; it also still reads the pre-0.2 header. The ``.json`` and
``.npy`` writers carry the values but not the :ref:`bit-depth tag
<doc-bitdepth>` -- a quantised LUT reloaded from them comes back as float.

Hald as a format
----------------

A **Hald pattern** is the standard image for encoding a 3-D LUT in a single
picture (the pattern that ``ImageMagick`` calls ``hald:N``): an ``N*N``-ish
square whose RGB value at each pixel is exactly one LUT input node, so the
picture is the LUT laid out as a texture. ``lutlab.formats.hald`` is a codec for
that pattern:

* **Write** renders the LUT onto a Hald grid and saves a PNG. The PNG is 8-bit
  or 16-bit, chosen from the LUT's :ref:`bit depth
  <doc-bitdepth>`; a quantised LUT is stored at its exact depth, so the
  round-trip is *lossless* (max error ``1 / (2**bit_depth - 1)``).
* **Read** parses the pattern back into a :class:`lutlab.LUT`, inferring the
  stored depth from the file.

.. code-block:: python

   lut.save("grade.png")          # a Hald chart, 8-bit for an 8-bit LUT
   lut2 = lutlab.LUT.load("grade.png")
   lut2.allclose(lut)             # True (lossless round-trip)

.. note::

   Writing a Hald chart needs OpenCV (``cv2``) and a cubic (3-D) LUT. Reading
   is the same. The pattern itself is pure numpy; only the PNG I/O needs
   ``cv2``, because PIL and imageio in current releases cannot write 16-bit
   *RGB* PNG.
