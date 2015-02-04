=======
pysimoa
=======

A Scientific Python package for Simulation Output Analysis



Requirements and Python 2/3 compatibility
-----------------------------------------

This package runs under **Python 2** and **Python 3**, and has been tested with
**Python 2.7.6** and **Python 3.4.0**.

License
-------

See the `LICENSE <LICENSE>`_ file.


Developing
----------

Development environment
~~~~~~~~~~~~~~~~~~~~~~~

Use `tox`_ to `prepare virtual environments for development`_.

.. _prepare virtual environments for development: http://testrun.org/tox/latest/example/devenv.html

.. _tox: http://tox.testrun.org

To set up a **Python 2.7** environment in ``.devenv27``, run::

    $ tox -e devenv27

To set up a **Python 3.4** environment in ``.devenv34``, run::

    $ tox -e devenv34

Packaging
~~~~~~~~~

This package uses `setuptools`_.

.. _setuptools: http://pythonhosted.org/setuptools

Run ::

    $ python setup.py sdist
   
or ::

    $ python setup.py bdist
   
or ::

    $ python setup.py bdist_wheel
    
to build a source, binary or wheel distribution.


Complete Git Integration
~~~~~~~~~~~~~~~~~~~~~~~~

Setuptool uses the information of tags to infer the version of your project
with the help of `versioneer <https://github.com/warner/python-versioneer>`_.

To use this feature you need to tag with the format ``MAJOR.MINOR[.REVISION]``
, e.g. ``v0.0.1`` or ``v0.1``.
The prefix ``v`` is needed!

Run ::
        
    $ python setup.py version
    
to retrieve the current `PEP440`_-compliant version.
This version will be used when building a package and is also accessible
through ``devs.__version__``.
The version will be ``unknown`` until you have added a first tag.

.. _PEP440: http://www.python.org/dev/peps/pep-0440

Pre-commit hooks
................

Unleash the power of Git by using its `pre-commit hooks
<http://pre-commit.com/>`_.

Run ::

    $ pre-commit install

to install the pre-commit hooks.

Sphinx Documentation
~~~~~~~~~~~~~~~~~~~~

Build the documentation with ::
        
    $ python setup.py docs
    
and run doctests with ::

    $ python setup.py doctest

Alternatively, let `tox`_
`configure the virtual environment and run sphinx <http://tox.readthedocs.org/en/latest/example/general.html#integrating-sphinx-documentation-checks>`_::

    $ tox -e docs

Add further options separated from tox options by a double dash ``--``::

    $ tox -e docs -- --help



Add `requirements`_ for building the documentation to the
`requirements-doc.txt <requirements-doc.txt>`_ file.

.. _requirements: http://pip.readthedocs.org/en/latest/user_guide.html#requirements-files

Continuous documentation building
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For continuously building the documentation during development, run::
        
    $ python setup.py autodocs

Unittest & Coverage
~~~~~~~~~~~~~~~~~~~

Run ::

    $ python setup.py test
    
to run all unittests defined in the subfolder ``tests`` with the help of `tox`_
and `py.test`_.

.. _py.test: http://pytest.org

The py.test plugin `pytest-cov`_ is used to automatically generate a coverage
report. 

.. _pytest-cov: http://github.com/schlamar/pytest-cov

Continuous testing
~~~~~~~~~~~~~~~~~~

For continuous testing in a **Python 2.7** environment, run::
       
    $ python2 setup.py test --tox-args='-c toxdev.ini -e py27'

For continuous testing in a **Python 3.4** environment, run::
       
    $ python3 setup.py test --tox-args='-c toxdev.ini -e py34'


Requirements Management
~~~~~~~~~~~~~~~~~~~~~~~

Add `requirements`_ to the `requirements.txt <requirements.txt>`_ file which
will be automatically used by ``setup.py``.
