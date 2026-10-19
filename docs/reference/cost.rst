Cost
====

.. _class-Cost:

The static cost a transform reports before any pixel is touched.

.. rst-class:: classref-introduction-group

Description
-----------

``lutlab.Cost`` is the value returned by a transform's ``cost()`` -- for a
:ref:`LUT <class-LUT>`, :ref:`LUT.cost() <class-LUT-cost>`. It
summarises what a transform costs **before** it touches a single pixel: how many
scalars it stores, how many bytes those take, and how many memory reads it does
per output pixel. It is a frozen dataclass, so it is safe to compare and hash.

.. rst-class:: classref-reftable-group

Fields
------

.. list-table::
   :header-rows: 1
   :widths: 22 22 16 40

   * - **Field**
     - **Type**
     - **Default**
     - **Description**
   * - :ref:`params <class-Cost-params>`
     - ``int``
     - ``-``
     - Learnable / stored scalars.
   * - :ref:`nbytes <class-Cost-nbytes>`
     - ``int``
     - ``-``
     - Storage of those scalars.
   * - :ref:`reads_per_px <class-Cost-reads_per_px>`
     - ``int | None``
     - ``None``
     - Memory reads per pixel, if meaningful.
   * - :ref:`meta <class-Cost-meta>`
     - ``dict[str, Any]``
     - ``{}``
     - Free-form extra metadata.

.. rst-class:: classref-reftable-group

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - **Returns**
     - **Method**
   * - ``dict``
     - :ref:`as_dict() <class-Cost-as_dict>`

----

.. rst-class:: classref-descriptions-group

Field descriptions
------------------

.. _class-Cost-params:

.. rst-class:: classref-property

**params** → ``int``

The number of learnable / stored scalars. For a :ref:`LUT
<class-LUT>` this is the number of grid points; for a :ref:`LUTMixture
<class-LUTMixture>` the sum over its LUTs.

----

.. _class-Cost-nbytes:

.. rst-class:: classref-property

**nbytes** → ``int``

The storage of those scalars in bytes. For a :ref:`LUT <class-LUT>` this is
:ref:`LUT.nbytes <class-LUT-nbytes>` -- compact when the LUT is quantised.

----

.. _class-Cost-reads_per_px:

.. rst-class:: classref-property

**reads_per_px** → ``int | None``

Memory reads performed per output pixel, for the interpolation scheme in use
(1 for nearest, ``2 ** n_in`` for multilinear). ``None`` when not meaningful.

----

.. _class-Cost-meta:

.. rst-class:: classref-property

**meta** → ``dict[str, Any]``

Free-form extra metadata. For example a :ref:`LUT <class-LUT>` reports
``{"sizes": [...]}`` and a :ref:`LUTMixture <class-LUTMixture>` reports
``{"n_luts": S}``.

----

.. rst-class:: classref-descriptions-group

Method descriptions
-------------------

.. _class-Cost-as_dict:

.. rst-class:: classref-method

**Cost.as_dict** ``() -> dict[str, Any]``

Return the fields as a plain ``dict`` (for logging and reporting).
