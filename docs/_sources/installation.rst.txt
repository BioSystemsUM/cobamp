
************
Installation
************

Basic requirements
==================

* Python 3.x
* `CPLEX <https://www.ibm.com/products/ilog-cplex-optimization-studio>`_ along with its Python wrapper installed in your current distribution (please note that the Python version must be compatible with CPLEX)

Optional requirements
=====================

For easier model loading and analysis (using constraint-based methods), the following libraries can be used:

* `cobrapy <https://github.com/opencobra/cobrapy>`_
* `framed <https://github.com/cdanielmachado/framed>`_

Additionally, the `escher <https://escher.github.io/>`_ library can be used to display elementary flux modes on metabolic maps and the `networkx <https://networkx.github.io/>`_ library can also be used to plot trees generated from EFM or MCS enumeration.

Via pip
=======

The easiest method is to use pip to `install the package from PyPI <https://pypi.python.org/pypi/cobra>`_::

    pip install metaconvexpy

From source
===========

* Download the latest source files from github
* Unpack the source files into a directory of your choosing
* Open the operating system's command-line interface
* Change into the source file directory
* Run the following command ::

    python setup.py install

It is highly recommended that this package along with its requirements are installed in a separate Python environment.
Tools such as `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ or `conda <https://conda.io/docs/>`_ can be used to create Python environments.

