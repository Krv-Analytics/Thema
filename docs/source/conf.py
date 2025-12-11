# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Thema"
copyright = "2024, Sidney Gathrid, Stuart Wayland, Jeremy Wayland"
author = "Sidney Gathrid, Stuart Wayland, Jeremy Wayland"
release = "v0.1.4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinxcontrib.mermaid",
    # 'sphinxcontrib.jupyter',  # Uncomment if using sphinxcontrib-jupyter
]

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
nbsphinx_execute = "always"
nbsphinx_kernel_name = "python3"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_theme = "alabaster"
# html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
html_title = "Thema v0.1.3"

html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo-dark.png",
    "sidebar_hide_name": False,
}


html_css_files = [
    "custom.css",
]

html_js_files = [
    "logo-link.js",
]

nbsphinx_allow_errors = True
