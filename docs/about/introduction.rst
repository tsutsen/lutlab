Introduction
============

``lutlab`` is a small library and testbed for the colour-LUT math that appears in
every research paper on LUT-based colour grading. It gives you one clean type for
a LUT, a handful of well-defined operations on it, and an evaluation harness so
that different methods produce directly comparable numbers.

What it does
------------

* **Load and save LUTs** in the common formats -- ``.cube``, ``.json``, ``.npy``
  -- and the standardised Hald pattern as a ``.png``.
* **Apply LUTs** to images on CPU or GPU, with several interpolation schemes
  (:ref:`nearest <scheme-nearest>`, trilinear, tetrahedral) and several
  backends (:ref:`backends <sec-backends>`).
* **Extract a LUT** back from a Hald render or a before/after image pair.
* **Measure results** with quality metrics (CIEDE2000, PSNR, SSIM, LPIPS),
  timing, and a small power model.

.. note::

   The emphasis is on *comparability*, not on doing everything. There is one
   implementation per operation, so a number measured with ``lutlab`` is
   comparable to any other number measured with ``lutlab``.

Why a testbed
-------------

A LUT-based method is only interesting relative to the baselines around it: the
identity LUT, a straight Hald extraction, a segmented fit, and a learned LUT.
``lutlab`` keeps all of those behind one interface -- the
:ref:`ColorTransform protocol <sec-colortransform>` -- so an experiment is "build
a transform, evaluate it, report it" rather than a pile of ad-hoc scripts that
cannot be re-run or compared.

.. seealso::

   * :ref:`Core concepts <doc-design>` -- the shape convention, backends, and
     input/output rules.
   * :ref:`Quickstart <doc-quickstart>` -- your first LUT in a few
     lines.
