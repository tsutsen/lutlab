Formats
=======

.. _mod-formats:

Load and save LUTs to and from file, by extension or explicit format.

.. rst-class:: classref-introduction-group

Description
-----------

``lutlab.formats`` reads and writes LUTs. The user-facing entry points are
:ref:`LUT.load <class-LUT-load>` and :ref:`LUT.save <class-LUT-save>`; the
functions here are the codec engines they call, dispatching on the file
extension (or an explicit ``format=``). Each format lives in its own submodule
with ``read`` / ``write`` (and, for the text formats, ``loads`` / ``dumps``).
The supported formats and their fidelity trade-offs are described in :ref:`File
formats <doc-formats>`.

.. rst-class:: classref-introduction-group

Tutorials
---------

- :ref:`File formats <doc-formats>`

- :ref:`Bit depth and compact storage <doc-bitdepth>`

.. rst-class:: classref-reftable-group

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - **Function**
     - **Description**
   * - :ref:`load <func-formats-load>`
     - Read a LUT from a file (``.cube`` / ``.npy`` / ``.json`` / ``.png``); the engine behind :ref:`LUT.load <class-LUT-load>`.
   * - :ref:`save <func-formats-save>`
     - Write a LUT to a file, choosing the format from the extension; the engine behind :ref:`LUT.save <class-LUT-save>`.
   * - :ref:`formats <func-formats-formats>`
     - List the supported file extensions.

.. rst-class:: classref-reftable-group

Submodules
----------

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - **Submodule**
     - **Extensions**
     - **Description**
   * - ``cube``
     - ``.cube``
     - Adobe Photoshop / 3D LUT ``.cube`` text format (8/16-bit, uniform grid).
   * - ``hald``
     - ``.png``
     - Hald CLUT pattern (8-bit or 16-bit PNG); round-trips ``bit_depth``.
   * - ``lutjson``
     - ``.json``
     - Lossless JSON container (base64 float32 grid) with axes / domain / space.
   * - ``npy``
     - ``.npy``
     - Raw float32 grid via ``numpy`` -- the simplest round-trip.

----

.. rst-class:: classref-descriptions-group

Function descriptions
---------------------

.. _func-formats-load:

.. rst-class:: classref-method

**load** ``(path: str, *, format: str | None = None, **kw) -> LUT``

Read a LUT from ``path``, inferring the format from the extension unless
``format=`` is given. This is what :ref:`LUT.load <class-LUT-load>` calls. See
:ref:`File formats <doc-formats>`.

----

.. _func-formats-save:

.. rst-class:: classref-method

**save** ``(lut: LUT, path: str, *, format: str | None = None, **kw) -> None``

Write a LUT to ``path``, choosing the format from the extension. This is what
:ref:`LUT.save <class-LUT-save>` calls. See :ref:`File formats
<doc-formats>`.

----

.. _func-formats-formats:

.. rst-class:: classref-method

**formats** ``() -> list[str]``

List the file extensions lutlab can read and write (for example
``[".cube", ".npy", ".json", ".png"]``).

See also :ref:`load() <func-formats-load>`, :ref:`save() <func-formats-save>`
and :ref:`File formats <doc-formats>`.
