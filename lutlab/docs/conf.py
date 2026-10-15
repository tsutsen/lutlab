# Sphinx config for the lutlab docs -- Read the Docs theme, the same one behind
# docs.godotengine.org. Build with:  python -m sphinx -b html docs OUT
import os
import sys

# Make `import lutlab` work from the build: put the package's parent (Experiments/)
# on sys.path (lutlab/__init__.py then wires up robust_hald / xlut itself).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

project = "lutlab"
author = "PhD LUT experiments"
copyright = "2026"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

# Read-the-Docs theme -- the Godot look.
html_theme = "sphinx_rtd_theme"
html_title = "lutlab"
html_short_title = "lutlab"
html_theme_options = {"collapse_navigation": False}

# robust_hald.flow.grid_flow's docstring uses |Sobel| as an absolute-value
# notation; render the pipes literally so the re-export page has no
# undefined-substitution errors.
rst_prolog = """
.. |Sobel| unicode:: 007C Sobel 007C
"""

# Autodoc: source order, `:members:` applied everywhere (so class methods show),
# original members only (re-export pages opt into :imported-members: per file).
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": False}
# Types in the call signature (the Godot look) and repeated in the parameter
# descriptions for the prose body.
autodoc_typehints = "both"

# Our docstrings are numpy-style ("Parameters\n----------"); napoleon renders them.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
