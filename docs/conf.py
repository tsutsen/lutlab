# -*- coding: utf-8 -*-
#
# lutlab documentation build configuration.
#
# Structured after the Godot Engine docs: RTD theme, one page per topic,
# captioned/hidden toctrees in ``index.rst``, and ``exclude_patterns`` that hide
# dotfiles so the source tree stays tidy in the generated output.

import os
from importlib.metadata import PackageNotFoundError, version as _package_version

project = "lutlab"
author = "the lutlab maintainers"
copyright = "2026, the lutlab maintainers"

# One source of truth: the installed package metadata (which hatch reads from
# ``lutlab/__init__.py``). Fall back only when building docs from a bare checkout.
try:
    release = _package_version("lutlab")
except PackageNotFoundError:
    release = "0.3.0"
version = ".".join(release.split(".")[:2])
needs_sphinx = "7.0"

# -- Build ------------------------------------------------------------------

master_doc = "index"
source_suffix = ".rst"
source_encoding = "utf-8-sig"

# Hide dotfiles and the build directory from the output.
exclude_patterns = [".*", "**/.*", "_build"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "signature"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- Styling ----------------------------------------------------------------

pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    # Show only the logo (no title text) next to it.
    "logo_only": False,
    # Do not collapse the navigation tree.
    "collapse_navigation": False,
    # Remove the version / language pickers beneath the title.
    "version_selector": False,
    "language_selector": False,
    # Keep the left nav attached to the page (Godot layout).
    "flyout_display": "attached",
}
html_static_path = ["_static"]
# Godot-style skin: automatic light/dark via prefers-color-scheme (no manual toggle).
html_css_files = ["css/custom.css", "css/nav.css"]
html_js_files = ["nav.js"]
html_title = "lutlab documentation"
