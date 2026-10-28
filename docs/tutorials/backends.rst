Backends
========

.. _doc-backends:

.. _sec-backends:

A **backend** is one implementation of the apply kernel
``(volume, x, axes, scheme) -> out``. ``lutlab`` ships four; each declares what it
can handle (device, grid type, ``n_in``, schemes), and ``backend="auto"`` picks
the best one available for your data and machine.

numpy
-----

The reference implementation. Vectorised, correct for any grid and any
``n_in``, and the **oracle** every other backend is tested against in the test
suite. Single-threaded; it is the ground truth, not the fast path.

.. code-block:: python

   out = lut(img, backend="numpy")

numba
-----

The fast CPU path. Multi-threaded, JIT-compiled, but restricted to **uniform**
grids (a ``rectilinear`` LUT falls back to ``numpy``). This is what
``backend="auto"`` selects for a ``numpy`` image on a machine with numba
installed.

.. code-block:: python

   out = lut(img, backend="numba")

torch
-----

Runs on CPU, CUDA, ROCm, or MPS and is the only backend that is
**differentiable** -- the output is a ``torch`` tensor with a live gradient, so
you can train *through* a LUT. It accepts any grid and any ``n_in``. Pass a
``torch`` tensor in and you get one out.

.. code-block:: python

   import torch
   timg = torch.rand(1, 720, 1280, 3)          # or .cuda()
   tout = lut(timg, backend="torch", interp="trilinear")

fused
-----

.. _backend-fused:

An opt-in, hand-written HIP/CUDA kernel compiled into lutlab for the common case: a
**cubic, uniform** 3-D LUT on a GPU, with nearest, multilinear and
tetrahedral interpolation. It is not selected by ``auto``;
you must ask for it by name. It needs ``torch`` + a CUDA/ROCm device (and the
matching nvcc/hipcc toolchain). It is *not* differentiable.

.. code-block:: python

   out = lut(img_cuda, backend="fused")        # img_cuda on the GPU

.. note::

   The fused kernel works on the raw GPU buffer end to end. The ``torch``
   backend is the general one -- use it when you need a non-cubic grid or a
   gradient.

Adding a backend
----------------

A backend is a function plus a capability check, registered into the same
registry the built-ins use. The test suite then validates it against the oracle
for free:

.. code-block:: python

   from lutlab.kernels import Backend, register

   def run(volume, x, axes, scheme):
       ...                                      # return [N, 3]
   register(Backend(name="mine", run=run,
                    unsupported=lambda req: None, priority=25))
