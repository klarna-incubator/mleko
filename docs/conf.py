"""Sphinx configuration."""
project = "ML-Ekosystem"
author = "Erik Båvenstrand"
copyright = "2023, Erik Båvenstrand"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
]
autodoc_typehints = "both"
html_theme = "furo"
autoapi_template_dir = "_templates/autoapi"
autoapi_dirs = ["../mleko"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
