"""Sphinx configuration."""
project = "ML-Ekosystem"
author = "Erik Båvenstrand"
copyright = "2023, Erik Båvenstrand"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
