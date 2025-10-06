import os, sys
from datetime import datetime
from importlib.metadata import version as pkg_version, PackageNotFoundError

# путь к твоему пакету (src-layout)
sys.path.insert(0, os.path.abspath("../src"))

project = "IduEdu"
author = "Donny"

try:
    release = pkg_version("iduedu")
except PackageNotFoundError:
    release = "0.0.0"

version = ".".join(release.split(".")[:2])   # короткая X.Y

copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

html_theme = "furo"
html_static_path = ["_static"]

# Markdown + notebooks
myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
nb_execution_mode = "off"
nb_render_image_options = {"align": "center"}

# Autodoc / autosummary
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Интеракции с внешней докой
intersphinx_mapping = {
    "python":    ("https://docs.python.org/3", None),
    "numpy":     ("https://numpy.org/doc/stable/", None),
    "pandas":    ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "networkx":  ("https://networkx.org/documentation/stable/", None),
    "shapely":   ("https://shapely.readthedocs.io/en/stable/", None),
}


# Игнор
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
