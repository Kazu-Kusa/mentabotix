# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "mentabotix"
copyright = "2024, Whth"
author = "Whth"
release = "0.1.5.19"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = list(
    {
        "sphinx.ext.autodoc",
        "sphinx.ext.autosectionlabel",
        "sphinx.ext.autosummary",
        "sphinx.ext.duration",
        "sphinx.ext.extlinks",
        "sphinx.ext.napoleon",
        "sphinx.ext.doctest",
        "sphinx.ext.todo",
        "sphinx.ext.coverage",
        "sphinx.ext.mathjax",
        "sphinx.ext.githubpages",
        "sphinx.ext.viewcode",
    }
)
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
root = Path(__file__).parent.parent.parent
sys.path.extend([(root / "src").absolute().as_posix(), (root / "tests").absolute().as_posix()])
