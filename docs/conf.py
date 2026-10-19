project = "lutlab"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
html_theme = "sphinx_rtd_theme"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["torch", "numba", "skimage", "imageio", "scipy", "lpips"]
