dtoolAI - reproducible deep learning
====================================

.. image:: https://badge.fury.io/py/dtoolai.svg
   :target: https://badge.fury.io/py/dtoolai
   :alt: PyPi package

dtoolAI is a library for supporting reproducible deep learning.

Installation
------------

dtoolAI requires Python version 3. It can be installed through pip:

.. code-block:: bash

    pip install dtoolai

You can also download this repository and install through:

.. code-block:: bash

    python setup.py install

dtoolAI requires Pytorch - installing this under Windows can be trickier, see
the `Pytorch getting started guide <https://pytorch.org/get-started/locally/>`_
for details. You may find this install easier with `conda <https://docs.conda.io>`_, e.g.:

.. code-block:: bash

   conda install pytorch torchvision -c pytorch

Documentation
-------------

Primary documentation: https://dtoolai.readthedocs.io/en/latest/

Detailed examples of API use are provided in the notebooks/ directory in this
repository.

Tests
-----

Running the tests requires pytest.

To run the faster tests in the test suite, use:

.. code-block:: bash

    pytest tests/ -m "not slow"

The test suite also includes full end-to-end tests that create datasets, train
models and evaluate them on those datasets. These are much slower, to run them
use:

.. code-block:: bash

    pytest tests/

