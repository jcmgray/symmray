# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

sys.path.append(os.path.abspath("./_pygments"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "symmray"
copyright = f"2022-{datetime.date.today().year}, Johnnie Gray"
author = "Johnnie Gray"

try:
    from symmray import __version__

    release = __version__
except ImportError:
    try:
        from importlib.metadata import version as _version

        release = _version("symmray")
    except ImportError:
        release = "0.0.0+unknown"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
]

nb_execution_mode = "off"
myst_heading_anchors = 4
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
autosectionlabel_prefix_document = True

# sphinx-autoapi
autoapi_dirs = ["../symmray"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2da193",
        "color-brand-content": "#2da193",
    },
    "dark_css_variables": {
        "color-brand-primary": "#65cabb",
        "color-brand-content": "#65cabb",
    },
    "light_logo": "logo-title.png",
    "dark_logo": "logo-title.png",
}

html_css_files = ["my-styles.css"]
html_static_path = ["_static"]
html_favicon = "_static/symmray-icon.ico"

pygments_style = "_pygments_light.MarianaLight"
pygments_dark_style = "_pygments_dark.MarianaDark"


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    import inspect

    import symmray

    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(symmray.__file__))

    if "+" in symmray.__version__:
        return (
            f"https://github.com/jcmgray/symmray/blob/"
            f"HEAD/symmray/{fn}{linespec}"
        )
    else:
        return (
            f"https://github.com/jcmgray/symmray/blob/"
            f"v{symmray.__version__}/symmray/{fn}{linespec}"
        )


extlinks = {
    "issue": ("https://github.com/jcmgray/symmray/issues/%s", "GH %s"),
    "pull": ("https://github.com/jcmgray/symmray/pull/%s", "PR %s"),
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "autoray": ("https://autoray.readthedocs.io/en/latest/", None),
}
