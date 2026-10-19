lutlab
======

``lutlab`` is a small, self-contained library for colour LUTs: load and save
LUTs, apply them on CPU or GPU with several interpolation schemes, extract a
LUT back from a rendered image, and measure the results under one evaluation
harness. One implementation per operation, so the numbers are directly
comparable.

If you are new here, start with the :doc:`quickstart
<getting_started/quickstart>` to go from nothing to a graded image in a couple
of lines.

Choose the tile that fits where you are:

.. raw:: html

    <style>
        .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 0.375rem;
            grid-template-areas:
            "new apply"
            "go-deeper contribute";
        }
        .grid-item {
            padding: 1rem;
            border-radius: 0.375rem;
            text-align: center;
        }
        .grid-item, .grid-item:visited { color: hsla(0, 0%, 100%, 0.9); }
        .grid-item:hover, .grid-item:focus { text-decoration: none; filter: brightness(120%); }
        .grid-item:active { filter: brightness(80%); }
        .new        { grid-area: new;        background-color: #166534; }
        .apply      { grid-area: apply;      background-color: #115e59; }
        .go-deeper  { grid-area: go-deeper;  background-color: #1e3a8a; }
        .contribute { grid-area: contribute; background-color: #831843; }
    </style>
    <div class="grid-container">
        <a class="grid-item new" href="about/introduction.html">
            I have never used a 3D LUT before,<br>
            <strong>I want to understand what one is.</strong>
        </a>
        <a class="grid-item apply" href="getting_started/quickstart.html">
            I know what a LUT is,<br>
            <strong>I want to apply one to an image.</strong>
        </a>
        <a class="grid-item go-deeper" href="tutorials/backends.html">
            I apply LUTs,<br>
            <strong>I want to compare backends and schemes.</strong>
        </a>
        <a class="grid-item contribute" href="community/contributing.html">
            I know lutlab,<br>
            <strong>I want to contribute.</strong>
        </a>
    </div>
    <br>

Get involved
------------

``lutlab`` is an open source project. The docs can always use help: if something
is unclear, out of date, or missing, open an issue or a pull request on the
`GitHub repository <https://github.com/tsutsen/lutlab>`_ and we will sort it out.
See :doc:`Contributing <community/contributing>` for how the docs are written and
built.

.. Below is the main table-of-contents tree. It is hidden on the page itself but
   makes up the sidebar for navigation.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: About
   :name: sec-about

   about/introduction
   about/design

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started
   :name: sec-getting-started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials
   :name: sec-tutorials

   tutorials/applying_a_lut
   tutorials/interpolation_schemes
   tutorials/backends
   tutorials/file_formats
   tutorials/bit_depth
   tutorials/measuring_quality

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference
   :name: sec-reference

   reference/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Community
   :name: sec-community

   community/contributing
   community/style_guide
