Style guide
===========

.. _doc-style:

The lutlab docs follow the structure and conventions of the Godot Engine
documentation, adapted for a Python library. Keep new pages consistent with
these.

Structure
---------

* Top-level sections: ``about/``, ``getting_started/``, ``tutorials/``,
  ``reference/``, ``community/``.
* ``docs/index.rst`` is the front page: a short welcome, profile tiles, and one
  hidden, captioned ``.. toctree::`` per section. The sidebar is built entirely
  from those toctrees.
* One topic per page. File names are ``snake_case.rst``.
* Reference pages are generated from the source with ``automodule`` /
  ``autoclass``; keep the hand-written part to a short "Description" intro.

RST conventions
---------------

* **Titles** are underlined with ``=``; **sections** with ``-``.
* **Cross-references:** use ``:doc:`Title <path>`` to link to a page and
  ``:ref:`Title <label>`` to link to a labelled point inside a page. Put the
  label (``.. _label:``) directly above the heading it names.
* **Identifiers** in prose are bold + double backticks: ``**a :class:`lutlab.LUT`**``
  or, for module members, the ``:class:`` / :func: role where a real target
  exists.
* **Code** goes in a ``.. code-block:: python`` block with a language.
* **Call-outs** use the standard admonitions: ``.. note::``, ``.. warning::``,
  ``.. tip::``, ``.. seealso::``.
* **Tables** for tabular data (arguments, backends, formats) are grid tables.
* ``.. rst-class::`` may be used to hook CSS defined in ``docs/_static/custom.css``
  (e.g. the API summary tables).

Tone
----

* Second person ("your image"), present tense.
* Prefer the shortest sentence that carries the meaning.
* One idea per sentence where possible; lead with the noun.
