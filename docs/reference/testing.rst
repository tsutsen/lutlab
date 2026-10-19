Testing
=======

.. _mod-testing:

Utilities for exercising and verifying transforms across backends.

.. rst-class:: classref-introduction-group

Description
-----------

``lutlab.testing`` is a small harness for checking that transforms behave
sensibly: generate a random LUT, verify every :ref:`backend <doc-backends>`
agrees, and run the full :ref:`conformance <class-testing-conformance>` suite
that every ``ColorTransform`` must pass. It is what the lutlab test-suite uses,
and it is the same API you should call to validate a custom transform.

.. rst-class:: classref-introduction-group

Tutorials
---------

- :ref:`Backends <doc-backends>`

- :ref:`Measuring quality <doc-quality>`

.. rst-class:: classref-reftable-group

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - **Function**
     - **Description**
   * - :ref:`random_lut <func-testing-random_lut>`
     - A reproducible, roughly-identity LUT for tests.
   * - :ref:`check_backends <func-testing-check_backends>`
     - Apply a LUT with every backend and report the max pairwise deviation.
   * - :ref:`conformance <class-testing-conformance>`
     - The checks every transform must pass; returns failure messages.
   * - :ref:`selftest <func-testing-selftest>`
     - Run the built-in self-test over the installed backends.

----

.. rst-class:: classref-descriptions-group

Function descriptions
---------------------

.. _func-testing-random_lut:

.. rst-class:: classref-method

**random_lut** ``(size: int = 17, n_in: int = 3, *, seed: int = 0, strength: float = 0.08, axes: tuple[ndarray] | None = None, name: str = "random") -> LUT``

Generate a reproducible LUT that is close to the identity (bounded by
``strength``) -- a convenient fixture for exercising backends and metrics
without a real grade.

----

.. _func-testing-check_backends:

.. rst-class:: classref-method

**check_backends** ``(lut: LUT, *, n: int = 500, seed: int = 0, atol: float = 1e-5) -> dict``

Apply ``lut`` at ``n`` random points with every registered :ref:`backend
<doc-backends>` and return the maximum pairwise deviation per scheme. A clean
way to confirm the backends agree.

----

.. _class-testing-conformance:

.. rst-class:: classref-method

**conformance** ``(transform, *, n: int = 256, seed: int = 0, resample_grid_size: int | None = None, resample_tol: float = 1e-4, resample_kw: dict | None = None, skip_resample: bool = False) -> list[str]``

Run the full suite of checks every transform must pass (output shape,
determinism, finite values, and that ``resample`` reproduces the native output
within ``resample_tol``). Returns a list of failure messages -- ``[]`` means the
transform conforms.

----

.. _func-testing-selftest:

.. rst-class:: classref-method

**selftest** ``() -> bool``

Run the built-in self-test over the currently installed :ref:`backends
<doc-backends>` (backend agreement + conformance on a random LUT) and report
whether everything passed.
