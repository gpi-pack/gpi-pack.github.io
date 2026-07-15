Installation
============

Requirements
------------

The current package requires Python 3.9 or later. The core installation
includes the dependencies for text, image, static, and dynamic inference.
Video file input/output and hyperparameter tuning use optional dependencies,
as described below.

From PyPI (Version 0.2.1)
-------------------------

**gpi_pack 0.2.1** is available from PyPI. Install or upgrade the core package
with:

.. code-block:: bash

   python -m pip install --upgrade gpi-pack

To include video support, Optuna tuning, or both, install the corresponding
extras:

.. code-block:: bash

   # Video reading and H.264 writing
   python -m pip install --upgrade "gpi-pack[video]"

   # Hyperparameter tuning
   python -m pip install --upgrade "gpi-pack[tune]"

   # Both optional feature sets
   python -m pip install --upgrade "gpi-pack[video,tune]"

Version 0.2.1 includes the static interfaces, video processing, scalar and
repeated-outcome dynamic estimators, and both optional dependency groups.

From Source
-----------

For an editable development installation, clone `the source repository
<https://github.com/gpi-pack/gpi_pack>`_ and install it from the repository
root:

.. code-block:: bash

   git clone https://github.com/gpi-pack/gpi_pack.git
   cd gpi_pack
   python -m pip install -e .

Use ``-e ".[video]"``, ``-e ".[tune]"``, or ``-e ".[video,tune]"`` to add
the optional dependencies. You can confirm the installed package version
with:

.. code-block:: python

   import gpi_pack

   print(gpi_pack.__version__)  # 0.2.1
