Installation
============

``lutlab`` needs only **numpy** for the core: the reference backend, the file
formats, and the :class:`lutlab.LUT` type. Everything else is an optional extra.

.. code-block:: bash

   pip install lutlab                 # core: numpy only
   pip install "lutlab[fast]"         # + numba CPU kernels
   pip install "lutlab[torch]"        # + differentiable / GPU kernels
   pip install "lutlab[png]"          # + OpenCV, for Hald .png read/write
   pip install "lutlab[predictor]"    # + torchvision, for the ResNet predictor
   pip install "lutlab[metrics]"      # + CIEDE2000 / SSIM
   pip install "lutlab[lpips]"        # + LPIPS (needs torch)
   pip install "lutlab[eval]"         # adds imageio for dataset folders

.. note::

   You can install your own ``torch`` (CUDA or ROCm) instead of the ``[torch]``
   extra -- ``lutlab`` just needs a working ``torch`` on the path.

Check what is actually available on your machine:

.. code-block:: bash

   python -m lutlab info

That prints the installed backend status (which backends are available, and
why the others are not) and the registered methods, so you know which of the
:ref:`backends <doc-backends>` paths will run here.

Developing
----------

.. code-block:: bash

   pip install -e ".[dev,test]"
   pytest

The test suite is parameterised over backends, so a new backend is automatically
checked against the ``numpy`` oracle and against ``scipy`` where relevant.
