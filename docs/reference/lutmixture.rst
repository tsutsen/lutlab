LUTMixture
==========

.. _class-LUTMixture:

A weighted blend of LUTs, ``y = sum_s w_s * LUT_s(x)``.

.. rst-class:: classref-introduction-group

Description
-----------

``LUTMixture`` holds ``n_luts`` LUTs on the same grid and blends their outputs
with weights of shape ``(n_luts,)`` (global) or ``(..., n_luts)`` (per-pixel
masks). Weights come in through ``context=`` at call time, or are bound at
construction with ``weights=``. It is the segmented-grade generalisation of a
single :ref:`LUT <class-LUT>`: instead of one volume, it holds several and mixes
them. Collapse the blend to one dense LUT with :ref:`to_lut()
<class-LUTMixture-to_lut>`.

.. rst-class:: classref-introduction-group

Tutorials
---------

- :ref:`Backends <doc-backends>`

- :ref:`Applying a LUT <doc-applying>`

- :ref:`Measuring quality <doc-quality>`

.. rst-class:: classref-reftable-group

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 16 24 14 46

   * - **Member**
     - **Type**
     - **Default**
     - **Description**
   * - :ref:`luts <class-LUTMixture-luts>`
     - ``list[LUT]``
     - ``-``
     - The constituent LUTs (all on the same grid).
   * - :ref:`n_luts <class-LUTMixture-n_luts>`
     - ``int``
     - ``-``
     - Number of LUTs (``len(luts)``).
   * - :ref:`weights <class-LUTMixture-weights>`
     - ``ndarray | None``
     - ``None``
     - Optionally bound weights; otherwise passed as ``context=``.
   * - :ref:`name <class-LUTMixture-name>`
     - ``str``
     - ``"mixture"``
     - Human-readable label.
   * - :ref:`n_in <class-LUTMixture-n_in>`
     - ``int``
     - ``-``
     - Number of input channels (same for every LUT).

.. rst-class:: classref-reftable-group

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - **Returns**
     - **Method**
   * - :ref:`LUTMixture <class-LUTMixture>`
     - :ref:`LUTMixture <class-LUTMixture-__init__>`
   * - ``ndarray | Tensor``
     - :ref:`__call__ <class-LUTMixture-__call__>`
   * - :ref:`Cost <class-Cost>`
     - :ref:`cost <class-LUTMixture-cost>`
   * - :ref:`LUT <class-LUT>`
     - :ref:`to_lut <class-LUTMixture-to_lut>`
   * - :ref:`LUT <class-LUT>`
     - :ref:`resample <class-LUTMixture-resample>`

----

.. rst-class:: classref-descriptions-group

Property descriptions
---------------------

.. _class-LUTMixture-luts:

.. rst-class:: classref-property

**luts** → ``list[LUT]``

The constituent LUTs. Every LUT must have the same ``n_in`` (checked at
construction).

----

.. _class-LUTMixture-n_luts:

.. rst-class:: classref-property

**n_luts** → ``int``

The number of LUTs, ``len(luts)``.

----

.. _class-LUTMixture-weights:

.. rst-class:: classref-property

**weights** → ``ndarray | None``

Optionally bound weights. When ``None``, weights must be supplied as ``context=``
on every call.

----

.. _class-LUTMixture-name:

.. rst-class:: classref-property

**name** → ``str``

A human-readable label (default ``"mixture"``).

----

.. _class-LUTMixture-n_in:

.. rst-class:: classref-property

**n_in** → ``int``

Number of input channels, shared by every LUT in the mixture.

----

.. rst-class:: classref-descriptions-group

Method descriptions
-------------------

.. _class-LUTMixture-__init__:

.. rst-class:: classref-method

**LUTMixture** ``(luts: Sequence[LUT], name: str = "mixture", weights: ndarray | None = None) -> LUTMixture``

Create a mixture from a non-empty sequence of LUT objects (all with the same
``n_in``). ``weights`` is optional and may be bound here or passed later as
``context=``.

----

.. _class-LUTMixture-__call__:

.. rst-class:: classref-method

**LUTMixture.__call__** ``(x: ndarray | Tensor, *, context=None, interp: str = "multilinear", backend: str = "auto", device=None, check_finite: bool | None = None) -> ndarray | Tensor``

Apply every LUT to ``x`` and blend the outputs with the weights from ``context=``
(shape ``(n_luts,)`` or ``(..., n_luts)``), falling back to the bound
:ref:`weights <class-LUTMixture-weights>`. Global weights on a shared grid are
baked into one LUT first; per-pixel weights use a one-pass numpy/torch kernel,
while the numba and fused paths keep the per-LUT loop. See
:ref:`Backends <doc-backends>`.

----

.. _class-LUTMixture-cost:

.. rst-class:: classref-method

**LUTMixture.cost** ``() -> Cost``

The sum of the constituent LUTs' costs, with ``meta["n_luts"]`` set to
:ref:`n_luts <class-LUTMixture-n_luts>`. See :ref:`Cost <class-Cost>`.

----

.. _class-LUTMixture-to_lut:

.. rst-class:: classref-method

**LUTMixture.to_lut** ``(grid_size: int | None = None, *, weights: ndarray | None = None, **_kw) -> LUT``

Collapse the blend to one dense LUT on a uniform ``grid_size`` grid, using global
``(n_luts,)`` weights (exact when the LUTs share a grid and an input ``domain``;
different domains raise).

----

.. _class-LUTMixture-resample:

.. rst-class:: classref-method

**LUTMixture.resample** ``(grid_size: int | None = None, **kw) -> LUT``

The same as :ref:`to_lut <class-LUTMixture-to_lut>` for a mixture with bound
global weights. Per-pixel weights cannot be frozen into one LUT; pass global
``weights=`` or bind them at construction.
