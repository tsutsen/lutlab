Quickstart
==========

.. _doc-quickstart:

The shortest useful path through ``lutlab``: build a LUT, apply it to an image,
save and reload it, and score the result.

Apply a LUT
-----------

.. code-block:: python

   import numpy as np
   import lutlab

   # A synthetic 17^3 LUT, or load one you already have:
   lut = lutlab.testing.random_lut(17, seed=0)
   # lut = lutlab.LUT.load("grade.cube")

   img = np.random.rand(1080, 1920, 3).astype(np.float32)   # float in [0, 1]
   out = lut(img, interp="tetrahedral")                    # backend="auto"

   print(lutlab.metrics.e00(out, img).mean())              # mean CIEDE2000

``lut(img, ...)`` is the whole API. With ``backend="auto"`` it picks ``torch``
when the input is a torch tensor, gradients are requested, or the data is on a
GPU; otherwise ``numba`` if it is installed; otherwise the ``numpy`` oracle.
The output follows the input -- numpy in gives numpy out, a torch tensor in
gives a torch tensor out (and stays differentiable on ``torch``). The
``fused`` GPU kernel is opt-in: pass ``backend="fused"`` by name (see
:ref:`Backends <doc-backends>`).

Pick the interpolation with ``interp``: ``"trilinear"`` (the default),
``"nearest"``, or ``"tetrahedral"`` -- see
:ref:`Interpolation schemes <sec-schemes>`.

Save and reload
---------------

The extension picks the format. A Hald pattern is written as a ``.png``.

.. code-block:: python

   lut.save("grade.cube")     # .cube / .json / .npy / .png
   lut2 = lutlab.LUT.load("grade.cube")

   assert lut2.allclose(lut)

Where to go next
----------------

* :ref:`Applying a LUT <doc-applying>` -- interpolation schemes and
  how the backend is chosen.
* :ref:`File formats <doc-formats>` -- ``.cube``, ``.json``, ``.npy``
  and the Hald ``.png``.
* :ref:`Bit depth <doc-bitdepth>` -- the quality/size/speed lever.
* :ref:`Reference <doc-reference>` -- the full API.
