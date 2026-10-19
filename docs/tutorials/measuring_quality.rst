Measuring quality
=================

.. _doc-quality:

``lutlab.metrics`` splits measurement into three concerns: **quality**,
**timing**, and **power**. The evaluation harness ties them together.

Quality
-------

Compare a grade against its reference. These operate on the usual float ``[0, 1]``
images:

.. list-table::
   :header-rows: 1

   * - Function
     - Measures
     - Dependency
   * - :func:`lutlab.metrics.e00`
     - mean CIEDE2000 colour error
     - ``colour``
   * - :func:`lutlab.metrics.de76`
     - mean CIE76 error
     - ``colour``
   * - :func:`lutlab.metrics.psnr`
     - peak signal-to-noise ratio
     - numpy
   * - :func:`lutlab.metrics.ssim`
     - structural similarity
     - numpy
   * - :func:`lutlab.metrics.lpips`
     - perceptual (LPIPS) distance
     - ``lpips``
   * - :func:`lutlab.metrics.accuracy`
     - a combined quality score
     - --

.. code-block:: python

   import lutlab.metrics as m
   m.e00(out, ref).mean()       # mean CIEDE2000
   m.psnr(out, ref)             # PSNR in dB

``lut_space_accuracy`` measures a LUT's deviation in LUT space rather than in
image space -- useful when comparing a *method* that only produces a LUT.

Timing
------

.. list-table::
   :header-rows: 1

   * - Function
     - Measures
   * - :func:`lutlab.metrics.time_fn`
     - wall time of a callable
   * - :func:`lutlab.metrics.apply_speed`
     - apply throughput (pixels/s)
   * - :func:`lutlab.metrics.to_fps`
     - frame-rate for a given size

The :ref:`benchmark <doc-backends>` uses these to put a number next to
each :ref:`backend <doc-backends>`.

Power
------

:func:`lutlab.metrics.power` is a small energy model for an apply call (a
first-order estimate, not a power meter). It exists so a benchmark can report
an energy figure alongside speed without needing hardware.

Evaluation harness
------------------

``lutlab.metrics`` drives a *set* of methods over *a set* of data and returns
one comparable row per method. A ``lutlab.metrics.PairFolder`` is a
directory of before/after image pairs.

.. code-block:: python

   import lutlab.metrics as m

   folder = m.PairFolder("datasets/graded")
   rows = m.run_matrix(methods={...}, tasks={"graded": folder})   # one row per method
   for r in rows:
       print(r["method"], r["e00"], r["ms"])

See :ref:`the metrics reference <mod-metrics>` for the exact row fields
and how to add a dataset.
