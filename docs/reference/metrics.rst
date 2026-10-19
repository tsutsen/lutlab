Metrics
=======

.. _mod-metrics:

Fidelity and performance measurement over plain numpy arrays.

.. rst-class:: classref-introduction-group

Description
-----------

``lutlab.metrics`` measures how good and how fast a result is. It splits into
:ref:`quality <doc-quality>` (colour difference, PSNR, SSIM, LPIPS),
:ref:`timing <doc-backends>` (milliseconds / fps / megapixels per call), power
sampling, and the :ref:`evaluation harness <func-metrics-run_matrix>` that scores
a transform over a paired-image dataset. Unless noted, the per-pixel functions
take plain numpy arrays; colour metrics expect values in ``[0, 1]``.

.. rst-class:: classref-introduction-group

Tutorials
---------

- :ref:`Measuring quality <doc-quality>`

- :ref:`Backends <doc-backends>`

.. rst-class:: classref-reftable-group

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - **Function**
     - **Description**
   * - :ref:`e00 <func-metrics-e00>`
     - Per-pixel CIEDE2000 between two ``(..., 3)`` arrays in ``[0, 1]``.
   * - :ref:`de76 <func-metrics-de76>`
     - Per-pixel CIE76 colour difference (L2 distance in CIELAB).
   * - :ref:`psnr <func-metrics-psnr>`
     - Peak signal-to-noise ratio in dB (``+inf`` when identical).
   * - :ref:`ssim <func-metrics-ssim>`
     - Mean SSIM over an ``(H, W, 3)`` pair.
   * - :ref:`lpips <func-metrics-lpips>`
     - Learned perceptual distance; needs the ``lpips`` extra + torch.
   * - :ref:`accuracy <func-metrics-accuracy>`
     - Image-space accuracy of ``out`` vs ``ref`` (both ``(H, W, 3)``).
   * - :ref:`lut_space_accuracy <func-metrics-lut_space_accuracy>`
     - How far a LUT sits from a reference LUT on the grid.
   * - :ref:`time_fn <func-metrics-time_fn>`
     - Time a zero-arg callable (GPU-aware, with warm-up + repeat).
   * - :ref:`to_fps <func-metrics-to_fps>`
     - Convert a per-call time in ms to frames per second.
   * - :ref:`apply_speed <func-metrics-apply_speed>`
     - Time ``apply``: returns ms, fps and megapixels per second.
   * - :ref:`gpu_power_watts <func-metrics-gpu_power_watts>`
     - Current GPU package power in watts (``rocm-smi``), if available.
   * - :ref:`evaluate <func-metrics-evaluate>`
     - Score one transform over a set of image pairs.
   * - :ref:`run_matrix <func-metrics-run_matrix>`
     - Score many transforms across many datasets; write a results table.

.. rst-class:: classref-reftable-group

Types
-----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Type**
     - **Description**
   * - :ref:`PairFolder <class-metrics-PairFolder>`
     - An iterable of ``(input, target)`` pairs from a folder tree.
   * - :ref:`SCHEMA <class-metrics-SCHEMA>`
     - Column order for the table that :ref:`run_matrix <func-metrics-run_matrix>` writes.

----

.. rst-class:: classref-descriptions-group

Function descriptions
---------------------

.. _func-metrics-e00:

.. rst-class:: classref-method

**e00** ``(a: ndarray, b: ndarray, *, space: str = "srgb") -> ndarray``

Per-pixel CIEDE2000 colour difference between two ``(..., 3)`` arrays of values
in ``[0, 1]``. The most perceptually faithful difference; see :ref:`Measuring
quality <doc-quality>`.

----

.. _func-metrics-de76:

.. rst-class:: classref-method

**de76** ``(a: ndarray, b: ndarray, *, space: str = "srgb") -> ndarray``

Per-pixel CIE76 colour difference -- the L2 distance in CIELAB. Cheaper than
:ref:`e00 <func-metrics-e00>` but less faithful in the blue region.

----

.. _func-metrics-psnr:

.. rst-class:: classref-method

**psnr** ``(a: ndarray, b: ndarray) -> float``

Peak signal-to-noise ratio in dB between two ``(..., 3)`` arrays. ``+inf`` when
they are identical.

----

.. _func-metrics-ssim:

.. rst-class:: classref-method

**ssim** ``(a: ndarray, b: ndarray) -> float``

Mean structural similarity over an ``(H, W, 3)`` image pair, in ``[0, 1]``.

----

.. _func-metrics-lpips:

.. rst-class:: classref-method

**lpips** ``(a: ndarray, b: ndarray, *, net: str = "alex") -> float``

Learned perceptual image patch similarity. Needs the ``lpips`` extra and torch;
see :ref:`Measuring quality <doc-quality>`.

----

.. _func-metrics-accuracy:

.. rst-class:: classref-method

**accuracy** ``(out: ndarray, ref: ndarray, *, space: str = "srgb") -> dict``

A bundle of image-space accuracy scores of ``out`` against a reference ``ref``
(both ``(H, W, 3)``): PSNR, SSIM and delta-E.

----

.. _func-metrics-lut_space_accuracy:

.. rst-class:: classref-method

**lut_space_accuracy** ``(lut: LUT, ref_lut: LUT | None = None, *, space: str | None = None) -> dict``

How far a LUT sits from a reference LUT on the grid -- the LUT-space analogue of
:ref:`accuracy <func-metrics-accuracy>`. Use it to score a resampled or learned LUT
against a target.

----

.. _func-metrics-time_fn:

.. rst-class:: classref-method

**time_fn** ``(fn: Callable, *, warmup: int = 3, repeat: int = 20, sync: Callable | None = cuda_sync) -> float``

Time a zero-argument callable, returning mean milliseconds. Performs warm-up and
repeats, and calls ``sync`` between iterations (defaults to a CUDA/HIP sync so
GPU work is accounted).

----

.. _func-metrics-to_fps:

.. rst-class:: classref-method

**to_fps** ``(ms: float) -> float``

Convert a per-call time in milliseconds to frames per second (``1000 / ms``).

----

.. _func-metrics-apply_speed:

.. rst-class:: classref-method

**apply_speed** ``(transform, x: ndarray, *, warmup: int = 3, repeat: int = 20, **apply_kw) -> dict``

Time ``lutlab.kernels.apply(transform, x, **apply_kw)``: returns mean ms, fps
and megapixels per second. See :ref:`Backends <doc-backends>`.

----

.. _func-metrics-gpu_power_watts:

.. rst-class:: classref-method

**gpu_power_watts** ``(device: str = "0") -> float | None``

Current GPU package power draw in watts, read via ``rocm-smi`` (AMD) or
``nvidia-smi`` (NVIDIA). ``None`` when no GPU / tool is available.

----

.. _func-metrics-evaluate:

.. rst-class:: classref-method

**evaluate** ``(transform, pairs: Iterable, *, metrics: tuple[str, ...] = ("psnr", "ssim", "e00")) -> dict``

Apply ``transform`` to every input in ``pairs`` and score it against the
corresponding target with the named :ref:`metrics <doc-quality>`, returning the
mean score per metric.

----

.. _func-metrics-run_matrix:

.. rst-class:: classref-method

**run_matrix** ``(methods: Mapping[str, object], tasks: Mapping[str, Iterable], *, out: str | None = None, seed: int = 0, meta: dict | None = None) -> dict``

Score every transform in ``methods`` across every dataset in ``tasks``, returning
a results matrix (and writing a table to ``out`` when given). This is the
driver behind the paper's benchmark tables.

----

.. rst-class:: classref-descriptions-group

Type descriptions
-----------------

.. _class-metrics-PairFolder:

.. rst-class:: classref-method

**PairFolder** ``(root: str, input_dir: str = "input", target_dir: str = "target", exts: tuple[str, ...] = ..., bit_depth: int | None = None)``

An iterable of ``(input, target)`` image pairs found under ``root``/``input_dir``
and ``root``/``target_dir``. ``bit_depth`` optionally quantises the loaded pairs
to a given depth before scoring (see :ref:`Bit depth and compact storage
<doc-bitdepth>`).

----

.. _class-metrics-SCHEMA:

.. rst-class:: classref-property

**SCHEMA** → ``tuple[str, ...]``

The column order for the results table that :ref:`run_matrix
<func-metrics-run_matrix>` writes.
