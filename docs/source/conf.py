# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NEDAS'
copyright = '2025, Yue Ying'
author = 'Yue Ying'

from NEDAS import __version__ as release
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_member_order = 'bysource'

autodoc_mock_imports = ["tensorflow", "torch", "opencv", "pygrib", "pyfftw", "numba", "mpi4py"]

# Suppress duplicate-attribute warnings from Napoleon + autodoc both documenting
# annotated class attributes (e.g. dataclass fields, ClassVar annotations).
napoleon_use_ivar = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_show_sphinx = False
html_logo = '../imgs/nedas_logo_banner.png'

html_static_path = ['_static']
