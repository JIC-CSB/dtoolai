dtoolAI - reproducible deep learning
====================================

dtoolAI is a library for supporting reproducible deep learning.

Installation
------------

dtoolAI can be installed

You can also 

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

