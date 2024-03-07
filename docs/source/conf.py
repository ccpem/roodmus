# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "Lumache"
copyright = "2023, Joel Greer(UKRI), Tom Burnley (UKRI),\
    Maarten Joosten (TU Delft), Arjen Jakobi (TU Delft)"
author = "Joel Greer, Maarten Joosten, Tom Burnley, Arjen Jakobi"

release = "0.2"
version = "0.0.28"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
