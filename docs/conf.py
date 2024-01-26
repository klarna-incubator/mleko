"""Sphinx configuration."""
project = "mleko"
author = "Erik Båvenstrand"
copyright = "2023, Klarna Bank AB"
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
]
autodoc_typehints = "both"
html_theme = "furo"
autoapi_template_dir = "../_templates/autoapi"
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
autoapi_member_order = "bysource"
