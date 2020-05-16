dtoolAI - reproducible deep learning
====================================

.. image:: https://badge.fury.io/py/dtoolai.svg
   :target: https://badge.fury.io/py/dtoolai
   :alt: PyPi package

dtoolAI is a library for supporting reproducible deep learning.

Installation
------------

Through ``pip``
~~~~~~~~~~~~~~~

dtoolAI requires Python version 3 and Pytorch.

.. warning:: Install Pytorch before installing dtoolAI. For information on how to install Pytorch this see the
             `Pytorch getting started guide <https://pytorch.org/get-started/locally/>`_
             for details.

Once Pytorch has been installed dtoolAI can be installed through pip:

.. code-block:: bash

    pip install dtoolai

Through ``conda``
~~~~~~~~~~~~~~~~~

You can also install dtoolAI through conda. To optionally create a conda environment in which to install
dtoolAI:

.. code-block:: bash

    conda create -n dtoolai
    conda activate dtoolai

Then you can install with:

.. code-block:: bash

    conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
    conda install dtoolcore dtool-http dtoolai -c dtool

With ``setup.py``
~~~~~~~~~~~~~~~~~

You can also download this repository and install through:

.. code-block:: bash

    python setup.py install

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

