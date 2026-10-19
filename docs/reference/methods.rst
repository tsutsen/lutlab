Methods
=======

.. _mod-methods:

The registry of things that *produce* a transform from data.

.. rst-class:: classref-introduction-group

Description
-----------

``lutlab.methods`` is a registry for anything that **produces** a *transform*
from data -- a learned per-image
grade, a global trained model, a segmentation routine. A method is a function
``build(**cfg) -> transform`` registered under a name with
:ref:`register <func-methods-register>`, then called with
:ref:`build <func-methods-build>`. Because they all share this shape, the
evaluation harness and the :ref:`conformance <class-testing-conformance>` tests
treat every method the same way, whatever it is.

.. rst-class:: classref-introduction-group

Tutorials
---------

- :ref:`Applying a LUT <doc-applying>`

.. rst-class:: classref-reftable-group

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Function**
     - **Description**
   * - :ref:`register <func-methods-register>`
     - Decorator: register a ``build(**cfg) -> transform`` under a name.
   * - :ref:`get <func-methods-get>`
     - Look up a method's :ref:`MethodSpec <class-methods-MethodSpec>`.
   * - :ref:`names <func-methods-names>`
     - Sorted list of registered method names.
   * - :ref:`available <func-methods-available>`
     - Map each method to ``None`` if usable, else the reason it is not.
   * - :ref:`build <func-methods-build>`
     - Build (and return) the transform for a registered method.

.. rst-class:: classref-reftable-group

Types and submodules
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - **Item**
     - **Description**
   * - :ref:`MethodSpec <class-methods-MethodSpec>`
     - A registered method: name, build callable, dependencies, doc.
   * - ``builtin``
     - The built-in methods (for example ``identity``).
   * - ``pg_ia_nilut``
     - The NILUT-style learned per-image method (predicts a LUT per image).

----

.. rst-class:: classref-descriptions-group

Function descriptions
---------------------

.. _func-methods-register:

.. rst-class:: classref-method

**register** ``(name: str, *, requires: tuple[str, ...] = ())``

Decorator that registers a ``build(**cfg) -> ColorTransform`` function under
``name``. ``requires`` lists optional dependencies; a method whose dependencies
are missing reports its reason via :ref:`available() <func-methods-available>`.

.. code-block:: python

   import lutlab.methods as methods

   @methods.register("my_method", requires=("torch",))
   def build(*, image=None, target=None, **cfg):
       ...  # return a ColorTransform

----

.. _func-methods-get:

.. rst-class:: classref-method

**get** ``(name: str) -> MethodSpec``

Look up a registered method by name. Raises ``ValueError`` (listing what is
registered) when the name is unknown.

----

.. _func-methods-names:

.. rst-class:: classref-method

**names** ``() -> list[str]``

The sorted list of all registered method names.

----

.. _func-methods-available:

.. rst-class:: classref-method

**available** ``() -> dict[str, str | None]``

Map each registered method to ``None`` when it can run, or to a short reason
(e.g. ``"missing: torch"``) when one of its dependencies is not installed.

----

.. _func-methods-build:

.. rst-class:: classref-method

**build** ``(name: str, **cfg) -> transform``

Build the transform for a registered method, passing ``cfg`` to its ``build``
function. Raises ``ImportError`` when the method's dependencies are missing.

----

.. _class-methods-MethodSpec:

.. rst-class:: classref-method

**MethodSpec** ``(name: str, build: Callable, requires: tuple[str, ...] = (), doc: str = "")``

A frozen dataclass describing one registered method. Its
``unavailable_reason()`` returns ``None`` when every dependency in ``requires``
is installed, else a ``"missing: ..."`` string.
