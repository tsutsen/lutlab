Core concepts
=============

.. _doc-design:

.. _sec-colortransform:

``lutlab`` has one central object -- the :class:`lutlab.LUT` -- and one central
protocol -- the :ref:`ColorTransform <sec-colortransform>` interface that every
transform (a LUT, a mixture of LUTs, or a learned model) implements so that the
evaluation harness can drive them uniformly.

The shape convention
--------------------

A LUT volume has shape ``(D0, D1, ..., D_{n-1}, 3)``: the first ``n`` axes are
the **input** dimensions and the last axis is the **output RGB**.

* ``n_in`` is the number of input dimensions (3 for the usual 3D LUT, 4 for a
  4D LUT whose 4th input is passed separately as ``context``).
* The output is always 3 channels (sRGB by default; see ``LUT.space``).
* A bare ``numpy`` array with this shape is accepted anywhere a :class:`lutlab.LUT`
  is expected; a ``LUT`` is the same volume plus a name and methods.

Axes may differ in size and may be **non-uniform** (pass ``axes=`` to give the
sample position of each node). A :class:`lutlab.LUT` is immutable and copies its
input, so mutating the array after construction does nothing.

Backends
--------

Application of a LUT is dispatched through a small backend registry. Each backend
implements the same ``(volume, x, axes, scheme) -> out`` kernel and declares what it
can handle; ``backend="auto"`` picks the best available one for the data.

.. list-table::
   :header-rows: 1

   * - Backend
     - Runs on
     - Grids
     - ``n_in``
     - ``interp``
     - Differentiable
   * - ``numpy`` (oracle)
     - CPU
     - any
     - any
     - all
     - no
   * - ``numba``
     - CPU, multi-threaded
     - uniform
     - any
     - all
     - no
   * - ``torch``
     - CPU/CUDA/ROCm/MPS
     - any
     - any
     - all
     - **yes**
   * - ``fused`` (opt-in)
     - CUDA / ROCm
     - cubic, uniform
     - 3
     - nearest / trilinear / tetrahedral
     - no

The ``numpy`` backend is the **oracle**: every other backend is tested against it
in the test suite, so a new backend is checked for correctness for free. The
:ref:`backends <doc-backends>` tutorial compares them in detail.

Input and output rules
----------------------

* **Inputs are float** in the LUT's ``domain`` (default ``[0, 1]`` on every
  channel). Normalise integer images to float in [0, 1] first.
* Out-of-range inputs are **clamped** on every backend (no negative indices, no
  wrap-around).
* **NaN/inf inputs raise** ``ValueError``; integer images raise ``TypeError``.
  The library prefers a loud failure over a silently-wrong grade.
* **The return type follows the input**: ``numpy`` in gives ``numpy`` out, a
  ``torch`` tensor in gives a ``torch`` tensor out (and stays differentiable on
  the ``torch`` backend).
* ``device=None`` means "where the data already is". Pass ``device="cuda"`` to
  take a ``numpy`` image through a GPU backend.

Cost model
----------

.. _sec-cost:

Every transform reports a :class:`lutlab.Cost` (parameter count, bytes stored,
reads per pixel) via its ``cost()`` method. This is what the benchmark and the
evaluation harness use to put a memory figure next to a speed figure -- and it is
where a :ref:`quantised LUT's <sec-bit-depth>` smaller footprint shows up.
