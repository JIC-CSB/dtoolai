dtoolAI - reproducible deep learning
====================================

.. image:: https://badge.fury.io/py/dtoolai.svg
   :target: https://badge.fury.io/py/dtoolai
   :alt: PyPi package

.. image:: https://anaconda.org/dtool/dtoolai/badges/version.svg
   :target: https://anaconda.org/dtool/dtoolai

dtoolAI is a library for supporting reproducible deep learning.


Quick start
-----------

If you'd like to see what dtoolAI can do without installing anything, two of the
Jupyter notebooks in this repository highlighting dtoolAI functions can be run
through Google Colab without any local software installation:

* `Training a character recognition network <https://colab.research.google.com/drive/1vqKmQFK2utX8Bn0LQ_6lx_xB56r3nnUA?usp=sharing>`_
* `Retraining a network on new data <https://colab.research.google.com/drive/1vYS90QH7pW-9PLGXD9CKNXtqiTT6o3O1?usp=sharing>`_

You'll need a Google account to run these, and when you load the notebooks,
click "Open in playground" to be able to execute code.

Installation
------------

Dependencies
~~~~~~~~~~~~

dtoolAI is dependent on the following Python packages:

* pytorch
* torchvision
* dtoolcore
* dtool-http
* click
* pillow

If you install dtoolAI with ``pip`` or ``conda`` as described below, these
dependencies will be installed automatically. If you wish to install manually,
you'll need to install these before installing dtoolAI.

For Windows users, we recommend installing pytorch and torchvision through
anaconda/conda. See the section below for details.

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

To install the dtool command line utilities, you'll need to use pip:

.. code-block:: bash
    
    pip install dtool
    
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

