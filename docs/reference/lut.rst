LUT
===

.. _class-LUT:

An immutable colour-lookup volume: the central object of ``lutlab``.

.. rst-class:: classref-introduction-group

Description
-----------

A ``lutlab.LUT`` is an immutable grid of RGB outputs that maps input colours to
output colours. It is a single dense lookup volume with no learned weights, so
it is trivial to store, stream, and apply on any :ref:`backend <doc-backends>`.

The volume has shape ``(D0, ..., Dn-1, 3)``: the first ``n`` axes are the input
channels (axis 0 = input channel 0) and the last axis is the RGB output, so
``volume[i0, ..., in-1]`` is the output for the input grid point ``(i0, ...,
in-1)``. Inputs outside the LUT's :ref:`domain <class-LUT-domain>` are rescaled
to the unit cube before lookup. See :ref:`Applying a LUT <doc-applying>` for the
full layout.

Values are stored as float32 by default. Pass ``bit_depth=`` to quantise to
``2 ** bit_depth`` levels, which shrinks the object (a quality/size/speed lever)
and stores it in a compact ``uint8`` / ``uint16`` buffer. See :ref:`Bit depth
and compact storage <doc-bitdepth>`.

A LUT is immutable: derive variants with :ref:`replace() <class-LUT-replace>` or
:ref:`to_bit_depth() <class-LUT-to_bit_depth>`, and resample onto a new grid with
:ref:`resample() <class-LUT-resample>`. It carries a :ref:`name <class-LUT-name>`, a
colour :ref:`space <class-LUT-space>`, and reports its static :ref:`cost
<class-Cost>`. For a weighted blend of several LUTs see
:ref:`LUTMixture <class-LUTMixture>`.

.. rst-class:: classref-introduction-group

Tutorials
---------

- :ref:`Applying a LUT <doc-applying>`

- :ref:`Interpolation schemes <sec-schemes>`

- :ref:`Backends <doc-backends>`

- :ref:`Bit depth and compact storage <doc-bitdepth>`

- :ref:`File formats <doc-formats>`

- :ref:`Measuring quality <doc-quality>`

.. rst-class:: classref-reftable-group

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 15 24 14 47

   * - **Member**
     - **Type**
     - **Default**
     - **Description**
   * - :ref:`volume <class-LUT-volume>`
     - ``ndarray`` (float32)
     - ``-``
     - Read-only ``[0, 1]`` grid, shape ``(D0, ..., Dn-1, 3)``.
   * - :ref:`name <class-LUT-name>`
     - ``str``
     - ``"lut"``
     - Human-readable label.
   * - :ref:`space <class-LUT-space>`
     - ``str``
     - ``"srgb"``
     - Colour space the values live in.
   * - :ref:`n_in <class-LUT-n_in>`
     - ``int``
     - ``-``
     - Number of input channels (``len(sizes)``).
   * - :ref:`sizes <class-LUT-sizes>`
     - ``tuple[int, ...]``
     - ``-``
     - Number of samples on each input axis.
   * - :ref:`bit_depth <class-LUT-bit_depth>`
     - ``int | None``
     - ``None``
     - Quantisation level, or ``None`` for full float.
   * - :ref:`shape <class-LUT-shape>`
     - ``tuple[int, ...]``
     - ``-``
     - ``sizes + (3,)``.
   * - :ref:`nbytes <class-LUT-nbytes>`
     - ``int``
     - ``-``
     - Bytes actually stored (compact when quantised).
   * - :ref:`grid_size <class-LUT-grid_size>`
     - ``int``
     - ``-``
     - Grid size when every axis is equal (else raises).
   * - :ref:`axes <class-LUT-axes>`
     - ``tuple[ndarray] | None``
     - ``None``
     - Per-axis sample positions, or ``None`` when uniform.
   * - :ref:`is_uniform <class-LUT-is_uniform>`
     - ``bool``
     - ``-``
     - ``True`` for a uniform grid (``axes is None``).
   * - :ref:`domain <class-LUT-domain>`
     - ``tuple[tuple, tuple]``
     - ``None``
     - Input range defined over (``.cube`` ``DOMAIN_MIN/MAX``).
   * - :ref:`has_default_domain <class-LUT-has_default_domain>`
     - ``bool``
     - ``-``
     - ``True`` when the domain is the unit cube.

.. rst-class:: classref-reftable-group

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - **Returns**
     - **Method**
   * - :ref:`LUT <class-LUT>`
     - :ref:`LUT <class-LUT-__init__>`
   * - :ref:`LUT <class-LUT>`
     - :ref:`identity <class-LUT-identity>` *(classmethod)*
   * - ``ndarray | Tensor``
     - :ref:`__call__ <class-LUT-__call__>`
   * - :ref:`Cost <class-Cost>`
     - :ref:`cost <class-LUT-cost>`
   * - :ref:`LUT <class-LUT>`
     - :ref:`resample <class-LUT-resample>`
   * - ``float``
     - :ref:`max_delta <class-LUT-max_delta>`
   * - ``ndarray``
     - :ref:`delta_from_identity <class-LUT-delta_from_identity>`
   * - ``(ndarray, ndarray)``
     - :ref:`tone_curve <class-LUT-tone_curve>`
   * - :ref:`LUT <class-LUT>`
     - :ref:`to_bit_depth <class-LUT-to_bit_depth>`
   * - :ref:`LUT <class-LUT>`
     - :ref:`replace <class-LUT-replace>`
   * - ``bool``
     - :ref:`allclose <class-LUT-allclose>`
   * - ``None``
     - :ref:`save <class-LUT-save>`
   * - :ref:`LUT <class-LUT>`
     - :ref:`load <class-LUT-load>` *(classmethod)*

----

.. rst-class:: classref-descriptions-group

Property descriptions
---------------------

.. _class-LUT-volume:

.. rst-class:: classref-property

**volume** → ``numpy.ndarray`` (float32)

The read-only grid of outputs in ``[0, 1]``. Always float32: a :ref:`quantised
<doc-bitdepth>` LUT is upcast on read so every :ref:`backend <doc-backends>`
interpolates the same float volume. The buffer is write-protected; derive a new
LUT with :ref:`replace() <class-LUT-replace>` instead of mutating it.

----

.. _class-LUT-name:

.. rst-class:: classref-property

**name** → ``str``

A human-readable label, carried through :ref:`file formats <doc-formats>` and
used by ``__repr__``. Defaults to ``"lut"``.

----

.. _class-LUT-space:

.. rst-class:: classref-property

**space** → ``str``

The colour space the volume values live in (default ``"srgb"``). Passed to
:ref:`quality metrics <doc-quality>` so delta-E is computed in the right space.

----

.. _class-LUT-n_in:

.. rst-class:: classref-property

**n_in** → ``int``

Number of input channels, i.e. ``len(sizes)``. A 4D LUT has ``n_in == 4``; the
extra context channel is supplied through ``context=`` when the LUT is called.

----

.. _class-LUT-sizes:

.. rst-class:: classref-property

**sizes** → ``tuple[int, ...]``

Number of samples on each input axis. Axes may differ in size (a rectilinear
grid).

----

.. _class-LUT-bit_depth:

.. rst-class:: classref-property

**bit_depth** → ``int | None``

The value quantisation: the stored values occupy ``2 ** bit_depth`` levels.
``None`` means full float32 precision. See :ref:`Bit depth and compact storage
<doc-bitdepth>`.

----

.. _class-LUT-shape:

.. rst-class:: classref-property

**shape** → ``tuple[int, ...]``

``sizes + (3,)`` -- the full shape of :ref:`volume <class-LUT-volume>`.

----

.. _class-LUT-nbytes:

.. rst-class:: classref-property

**nbytes** → ``int``

Bytes actually stored. Compact for a quantised LUT (``uint8`` for
``bit_depth <= 8``, ``uint16`` for ``> 8``); see :ref:`Bit depth and compact
storage <doc-bitdepth>`.

----

.. _class-LUT-grid_size:

.. rst-class:: classref-property

**grid_size** → ``int``

Convenience for the grid size when every axis is equal (a cubic grid). Raises
``ValueError`` if the axes differ -- use :ref:`sizes <class-LUT-sizes>` instead.

----

.. _class-LUT-axes:

.. rst-class:: classref-property

**axes** → ``tuple[ndarray, ...] | None``

Per-axis input sample positions (each in ``[0, 1]``, strictly increasing), or
``None`` for a uniform grid. See :ref:`Interpolation schemes <sec-schemes>` for
how non-uniform axes change the lookup.

----

.. _class-LUT-is_uniform:

.. rst-class:: classref-property

**is_uniform** → ``bool``

``True`` when :ref:`axes <class-LUT-axes>` is ``None`` (a uniform grid).

----

.. _class-LUT-domain:

.. rst-class:: classref-property

**domain** → ``((float, ...), (float, ...))``

The input range the LUT is defined over, as ``(min, max)`` tuples -- the
``.cube`` ``DOMAIN_MIN`` / ``DOMAIN_MAX``. Inputs are rescaled to the unit cube
before lookup. See :ref:`File formats <doc-formats>`.

----

.. _class-LUT-has_default_domain:

.. rst-class:: classref-property

**has_default_domain** → ``bool``

``True`` when :ref:`domain <class-LUT-domain>` is the unit cube ``([0] * n,
[1] * n)``.

----

.. rst-class:: classref-descriptions-group

Method descriptions
-------------------

.. _class-LUT-__init__:

.. rst-class:: classref-method

**LUT** ``(volume: ndarray, name: str = "lut", *, axes: tuple[ndarray] | None = None, domain: tuple[tuple, tuple] | None = None, space: str = "srgb", bit_depth: int | None = None) -> LUT``

Create a LUT from a ``(D0, ..., Dn-1, 3)`` volume. ``volume`` is copied and made
read-only; ``bit_depth`` quantises the values and switches to compact storage.

.. code-block:: python

   import numpy as np
   import lutlab

   volume = np.random.default_rng(0).random((17, 17, 17, 3)).astype("float32")
   lut = lutlab.LUT(volume, name="grade", bit_depth=8)   # compact 8-bit storage

See also :ref:`Applying a LUT <doc-applying>` and :ref:`Bit depth and compact
storage <doc-bitdepth>`.

----

.. _class-LUT-identity:

.. rst-class:: classref-method

**LUT.identity** ``(grid_size: int = 17, n_in: int = 3, *, sizes: tuple[int, ...] | None = None, axes: tuple[ndarray] | None = None, name: str = "identity", space: str = "srgb") -> LUT`` *(classmethod)*

The do-nothing LUT: output equals the first ``min(n_in, 3)`` input axes.
``LUT.identity(17)`` gives a uniform 17-per-axis identity grid -- the natural
baseline for :ref:`quality metrics <doc-quality>`. ``n_in=1`` yields a grey
ramp; a 2-input identity has no canonical form and raises. See :ref:`Measuring
quality <doc-quality>`.

----

.. _class-LUT-__call__:

.. rst-class:: classref-method

**LUT.__call__** ``(x: ndarray | Tensor, *, context=None, interp: str = "multilinear", backend: str = "auto", device=None, check_finite: bool | None = None) -> ndarray | Tensor``

Apply the LUT to ``x`` of shape ``(..., C)`` in the LUT's domain. The output is
``(..., 3)`` and follows the input type (numpy in → numpy out, torch in → torch
out). ``interp`` selects the :ref:`interpolation scheme <sec-schemes>` and
``backend`` the :ref:`backend <doc-backends>`. The engine is
:ref:`lutlab.kernels.apply <func-kernels-apply>`.

----

.. _class-LUT-cost:

.. rst-class:: classref-method

**LUT.cost** ``(scheme: str = "multilinear") -> Cost``

The static cost of the transform: stored params, :ref:`nbytes
<class-LUT-nbytes>`, and reads-per-pixel for ``scheme``. See :ref:`Cost
<class-Cost>`.

----

.. _class-LUT-resample:

.. rst-class:: classref-method

**LUT.resample** ``(grid_size: int | None = None, **_kw) -> LUT``

A dense `LUT` resampled onto a uniform ``grid_size``-per-axis grid, preserving
``bit_depth``. ``grid_size`` is an int (same on every axis) or one size per
axis. Returns ``self`` when it is already a uniform ``grid_size`` grid. See
:ref:`Applying a LUT <doc-applying>`.

----

.. _class-LUT-max_delta:

.. rst-class:: classref-method

**LUT.max_delta** ``() -> float``

The largest per-channel change the LUT makes at any grid point, measured
against the identity on the same grid. ``0.0`` means the LUT is (numerically)
the identity.

----

.. _class-LUT-delta_from_identity:

.. rst-class:: classref-method

**LUT.delta_from_identity** ``() -> ndarray``

``volume`` minus the identity LUT on the same grid -- the per-grid-point change
the LUT applies. The shape is the LUT's ``volume`` shape.

----

.. _class-LUT-tone_curve:

.. rst-class:: classref-method

**LUT.tone_curve** ``(n: int = 33) -> (ndarray, ndarray)``

The LUT's output sampled along the grey diagonal ``(t, t, ..., t)``: returns
``(t, curve)`` where ``t`` has shape ``(n,)`` and ``curve`` shape ``(n, 3)``.
Useful for plotting a LUT's lift/gamma/gain behaviour.

----

.. _class-LUT-to_bit_depth:

.. rst-class:: classref-method

**LUT.to_bit_depth** ``(n: int) -> LUT``

A copy quantised to ``n`` bits. Down-converting is a lossy round; up-converting
beyond the current depth adds no precision and warns. See :ref:`Bit depth and
compact storage <doc-bitdepth>`.

----

.. _class-LUT-replace:

.. rst-class:: classref-method

**LUT.replace** ``(*, volume: ndarray | None = None, name: str | None = None, axes="keep", domain="keep", space: str | None = None, bit_depth: int | None = None) -> LUT``

Return a copy with the given fields changed; ``"keep"`` / ``None`` leaves a
field as-is. This is the only way to "edit" a LUT, because LUTs are immutable.

----

.. _class-LUT-allclose:

.. rst-class:: classref-method

**LUT.allclose** ``(other: LUT, atol: float = 1e-06) -> bool``

``True`` when two LUTs share a grid and their values match within ``atol``.

----

.. _class-LUT-save:

.. rst-class:: classref-method

**LUT.save** ``(path: str, **kw) -> None``

Write the LUT to ``path``; the format is inferred from the extension
(``.cube``, ``.npy``, ``.json``, ``.png``). See :ref:`File formats
<doc-formats>`.

----

.. _class-LUT-load:

.. rst-class:: classref-method

**LUT.load** ``(path: str, **kw) -> LUT`` *(classmethod)*

Read a LUT from ``path`` (the inverse of :ref:`save() <class-LUT-save>`). See
:ref:`File formats <doc-formats>`.
