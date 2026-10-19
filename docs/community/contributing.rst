Contributing
============

Contributions are welcome: new backends, formats, methods, and docs are all
valued. This page is about the docs; for code, see the repository README.

Building the docs
-----------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   pip install -e .              # so autodoc can import lutlab
   cd docs
   make html                     # -> docs/_build/html
   open _build/html/index.html

The build uses the Read the Docs theme. ``docs/.readthedocs.yml`` and
``docs/requirements.txt`` mirror the local build so Read the Docs and a laptop
produce the same output.

Adding a page
-------------

1. Add the ``.rst`` file to the right section (``about/``, ``getting_started/``,
   ``tutorials/``, ``reference/`` or ``community/``).
2. Add it to the hidden ``.. toctree::`` for that section in ``docs/index.rst``
   (or, for a reference page, to the toctree in ``docs/reference/index.rst``).
3. Link to it from a neighbouring page with ``:doc:`` or ``:ref:`` so it is
   reachable from the reading path, not just the sidebar.

See the :ref:`style guide <doc-style>` for the exact conventions.

Contributing code
-----------------

* **Add a backend:** implement the apply kernel, register it with a capability
  check, and the test suite validates it against the ``numpy`` oracle for free.
  See :ref:`Backends <doc-backends>`.
* **Add a format:** add a ``lutlab/formats/<ext>.py`` and register the
  extension in ``lutlab.formats``.
* **Add a method:** register it with ``lutlab.methods`` so the harness can drive
  it.

Run the full suite before opening a PR:

.. code-block:: bash

   pip install -e ".[dev]"
   pytest
