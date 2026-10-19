Reference
=========

.. _doc-reference:

The complete lutlab API, in two groups: the **classes** (one page each) and the
**modules** that make up the library.

A *transform* is any object that is callable and has ``resample``, ``cost``,
``name`` and ``n_in`` -- there is no base class. :ref:`LUT <class-LUT>` is the
dense implementation, :ref:`LUTMixture <class-LUTMixture>` a weighted blend, and
:ref:`methods <mod-methods>` builds the learned ones. Everything
satisfying this shape can be applied, benchmarked and
:ref:`conformance-tested <class-testing-conformance>` the same way.

.. toctree::
   :hidden:

   lut
   lutmixture
   cost
   metrics
   kernels
   formats
   testing
   methods

Classes
-------

.. _class-reference:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - **Class**
     - **Description**
   * - :ref:`LUT <class-LUT>`
     - The immutable colour-lookup volume -- the central object of lutlab.
   * - :ref:`LUTMixture <class-LUTMixture>`
     - A weighted blend of LUTs, ``y = sum_s w_s * LUT_s(x)``.
   * - :ref:`Cost <class-Cost>`
     - The static cost a transform reports before any pixel is touched.

Modules
-------

.. _sec-modules:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - **Module**
     - **Description**
   * - :ref:`metrics <mod-metrics>`
     - Fidelity (delta-E, PSNR, SSIM, LPIPS), timing, and the evaluation harness (run a transform over a paired-image dataset).
   * - :ref:`kernels <mod-kernels>`
     - The interpolation kernels, the backend registry, and ``apply`` / ``interpolate`` (the engines behind ``lut(x)``).
   * - :ref:`formats <mod-formats>`
     - The file codecs (``.cube``, ``.npy``, ``.json``, ``.png``) behind :ref:`LUT.save <class-LUT-save>` / :ref:`LUT.load <class-LUT-load>`.
   * - :ref:`testing <mod-testing>`
     - Cross-backend checks and the transform conformance suite.
   * - :ref:`methods <mod-methods>`
     - The registry of things that produce a transform from data.
