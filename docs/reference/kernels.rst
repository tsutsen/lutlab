Kernels
=======

.. _mod-kernels:

The interpolation kernels, the backend registry, and the ``apply`` / ``interpolate`` engines.

.. rst-class:: classref-introduction-group

Description
-----------

``lutlab.kernels`` is where the actual lookups happen. A :ref:`backend
<doc-backends>` is a callable with a fixed shape -- ``run(volume, x, axes,
scheme) -> out`` -- registered in a small priority-ordered registry.
:ref:`lutlab.kernels.apply <func-kernels-apply>` and :ref:`interpolate <func-kernels-interpolate>`
select a backend and call it. Four backends ship with lutlab (``numpy``,
``numba``, ``torch``, and the opt-in native ``fused`` HIP kernel); add your own
with :ref:`register <func-kernels-register>`.

.. rst-class:: classref-introduction-group

Tutorials
---------

- :ref:`Backends <doc-backends>`

- :ref:`Interpolation schemes <sec-schemes>`

- :ref:`Applying a LUT <doc-applying>`

.. rst-class:: classref-reftable-group

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Function**
     - **Description**
   * - :ref:`apply <func-kernels-apply>`
     - Map ``(..., C)`` colours through any transform (the engine behind ``lut(x)``).
   * - :ref:`interpolate <func-kernels-interpolate>`
     - Interpolate a raw volume at grid points ``x`` (no transform wrapper).
   * - :ref:`backends <func-kernels-backends>`
     - List the registered backends and their capabilities.
   * - :ref:`register <func-kernels-register>`
     - Add a backend to the registry (see :ref:`Backends <doc-backends>`).
   * - :ref:`select <func-kernels-select>`
     - Pick the backend for a request, by name or auto-dispatch.
   * - :ref:`canonical_scheme <func-kernels-canonical_scheme>`
     - Resolve an interp name to a concrete scheme for ``n_in`` channels.

.. rst-class:: classref-reftable-group

Types
-----

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - **Type**
     - **Description**
   * - :ref:`Backend <class-Backend>`
     - A backend: ``name``, ``priority``, and a ``run(volume, x, axes, scheme)`` callable.
   * - :ref:`BackendInfo <class-BackendInfo>`
     - Read-only view of a registered backend (name, priority, ``auto``, ``reference``).
   * - :ref:`Request <class-Request>`
     - What a backend needs to run (volume, points, axes, scheme).
   * - :ref:`BackendError <class-BackendError>`
     - Raised when no suitable backend is available for a request.

----

.. rst-class:: classref-descriptions-group

Function descriptions
---------------------

.. _func-kernels-apply:

.. rst-class:: classref-method

**apply** ``(lut, x: ndarray | Tensor, *, context=None, interp: str = "multilinear", backend: str = "auto", device=None, check_finite: bool | None = None) -> ndarray | Tensor``

Map an array ``x`` of shape ``(..., C)`` through any ``ColorTransform`` (a
:ref:`LUT <class-LUT>`, a :ref:`LUTMixture <class-LUTMixture>`, a learned
method, or a bare ``(D..., 3)`` grid). This is the engine behind ``lut(x)``; the
``LUT`` and ``LUTMixture`` ``__call__`` methods and any learned method all
delegate to it. ``interp`` selects the :ref:`interpolation scheme <sec-schemes>`
and ``backend`` the :ref:`backend <doc-backends>`.

----

.. _func-kernels-interpolate:

.. rst-class:: classref-method

**interpolate** ``(volume: ndarray | Tensor, x: ndarray | Tensor, *, axes=None, scheme: str = "multilinear", backend: str = "auto", device=None, check_finite: bool | None = None) -> ndarray | Tensor``

Interpolate a raw ``(D0, ..., Dn-1, 3)`` ``volume`` at the grid points ``x`` of
shape ``(..., n_in)``. This is the low-level core that :ref:`lutlab.kernels.apply
<func-kernels-apply>` calls after domain rescaling. ``scheme`` is one of the
:ref:`interpolation schemes <sec-schemes>`; ``backend`` a
:ref:`backend <doc-backends>`.

----

.. _func-kernels-backends:

.. rst-class:: classref-method

**backends** ``() -> dict[str, BackendInfo]``

List the registered :ref:`backends <doc-backends>` as a mapping of name to a
:ref:`BackendInfo <class-BackendInfo>` (availability and dispatch flags). See
:ref:`Backends <doc-backends>` for how dispatch chooses among them.

----

.. _func-kernels-register:

.. rst-class:: classref-method

**register** ``(backend: Backend) -> None``

Add a backend to the registry. The backend's ``name`` and ``priority`` decide
where it sits in :ref:`dispatch <doc-backends>`; set ``auto=False`` to make it
opt-in only. See :ref:`Backends <doc-backends>` for a full example.

----

.. _func-kernels-select:

.. rst-class:: classref-method

**select** ``(req: Request, name: str = "auto") -> Backend``

Choose the backend that will service a :ref:`Request <mod-kernels>` -- by
explicit ``name`` or by auto-dispatch over the registered backends. Raises
``BackendError`` when nothing fits.

----

.. _func-kernels-canonical_scheme:

.. rst-class:: classref-method

**canonical_scheme** ``(mode: str, n_in: int) -> str``

Resolve an interp name (e.g. ``"linear"``, ``"tetrahedral"``, ``"tetrahedron"``)
to the concrete :ref:`scheme <sec-schemes>` used for ``n_in`` channels. The valid
names are listed in ``lutlab.kernels.SCHEMES``.

----

.. rst-class:: classref-descriptions-group

Type descriptions
-----------------

.. _class-Backend:

.. rst-class:: classref-method

**Backend** ``(name: str, run: Callable, *, priority: int = 0, reference: bool = False, auto: bool = True, ...)``

A backend: a ``name``, a ``run(volume, x, axes, scheme) -> out`` callable, and
dispatch metadata (``priority``, ``reference``, ``auto``). ``run`` must match
the :ref:`backend contract <doc-backends>`. Build one and hand it to
:ref:`register <func-kernels-register>`.

----

.. _class-BackendInfo:

.. rst-class:: classref-method

**BackendInfo** ``(name, available: bool, reason: str | None, differentiable: bool, reference: bool, auto: bool, priority: int)``

A read-only snapshot of a registered backend, as returned by
:ref:`backends() <func-kernels-backends>`: ``name``, whether it is
``available`` on this machine (with a ``reason`` when not), and its dispatch
flags.

----

.. _class-Request:

.. rst-class:: classref-method

**Request** ``(volume, scheme: str, n_in: int, sizes, uniform: bool, device: str, grad: bool = False, from_torch: bool = False)``

Everything a backend's ``run`` needs to know about one call: the ``volume``,
the lookup ``scheme``, the grid (``n_in``, ``sizes``, ``uniform``) and the
``device``. Passed to :ref:`select <func-kernels-select>`.

----

.. _class-BackendError:

.. rst-class:: classref-method

**BackendError**

Raised by :ref:`lutlab.kernels.apply <func-kernels-apply>` /
:ref:`lutlab.kernels.select <func-kernels-select>` when no registered backend
can service a :ref:`Request <class-Request>` (for example when every candidate
lacks a dependency). A subclass of ``ValueError``; re-exported as
``lutlab.BackendError``.
