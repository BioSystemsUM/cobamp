|License| |PyPI version|

pyCoBAMP
============

*pyCoBAMP* (Constraint-Based Analysis of Metabolic Pathways) is a Python package containing pathway analysis methods
for use with constraint-based metabolic models. The main purpose is to provide a framework that is both modular and
flexible enough to be integrated in other packages (such as cobrapy, framed or cameo) that already implement generic
data structures for metabolic models.

A (MI)LP solver is required for most of the methods. Current methods are implemented using CPLEX as a solver,
although future versions will use a unifying solver platform, such as optlang.

Current methods include:
   -  Elementary flux modes: K-Shortest algorithm
   -  Minimal cut sets: MCSEnumerator approach
   -  Elementary flux patterns: K-Shortest algorithm


Documentation
~~~~~~~~~~~~~



Instalation from PyPI (stable releases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    pip install cobamp

Instalation from github (latest development release)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    pip install https://github.com/skapur/pyCoBAMP/archive/master.zip

Credits and License
~~~~~~~~~~~~~~~~~~~

Developed at the Centre of Biological Engineering, University of Minho

Released under the GNU Public License (version 3.0).


.. |License| image:: https://img.shields.io/badge/license-GPL%20v3.0-blue.svg
   :target: https://opensource.org/licenses/GPL-3.0
.. |PyPI version| image:: https://badge.fury.io/py/pyCoBAMP.svg
   :target: https://badge.fury.io/py/pyCoBAMP
