# Configuration file for the Sphinx documentation builder.
import sys
import os
import datetime

# -- Project information -----------------------------------------------------
project = 'GPI: GenAI-Powered Inference'
copyright = f'{datetime.datetime.now().year}, Kentaro Nakamura, Kosuke Imai'
author = 'Kentaro Nakamura, Kosuke Imai'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    # 'sphinx.ext.viewcode',  # Disabled to remove "View page source" links
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]

# Explicitly set the master document
master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_baseurl = 'https://gpi-pack.github.io/'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None #"_static/images/gpi.png"  # Path to your logo file
html_favicon = "_static/images/gpi.png"  # Path to your logo file

# -- HTML theme settings -----------------------------------------------------
html_show_sourcelink = False  # Disable "View page source" link
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html'
    ]
}
# Force the global toctree to be included on every page
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': False,
    'titles_only': False,
    # Disable view source link
    'display_github': False,
    'viewcode_enable': False,
    # Show global navigation
    'globaltoc_collapse': False,
    'globaltoc_includehidden': False
}

html_css_files = [
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
    'custom.css',
]

html_js_files = [
    'js/copybutton.js',
]

sys.path.insert(0, os.path.abspath('.'))

# Add the citation extension to your extensions list
extensions = [
    'citation',  # Our custom citation extension
]

# Add these settings for GitHub Pages
html_context = {
    'display_github': False,
    'github_user': 'gpi-pack',
    'github_repo': 'gpi-pack.github.io',
    'github_version': 'main',  # or whatever your default branch is
    'conf_py_path': '/source/'
}