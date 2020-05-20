Installation
------------

dtoolAI requires Python version 3 and Pytorch.

.. warning:: Install Pytorch before installing dtoolAI. For information on how to install Pytorch this see the
             `Pytorch getting started guide <https://pytorch.org/get-started/locally/>`_
             for details. Version 1.4.0 of Pytorch and 0.5.0 of torchvision are definitely compatible with dtoolAI.

For Windows users, we'd recommend using ``conda`` to instal pytorch and torchvision, as per instructions below.

Installing with ``pip``
~~~~~~~~~~~~~~~~~~~~~~~

You can install dtoolAI via the pip package manager:

.. code-block:: bash

    pip install dtoolai

To understand the examples, it's also useful to install the `dtool meta package <https://dtool.readthedocs.io/>`_. This makes it easier to work with datasets created by dtoolAI:

.. code-block:: bash

    pip install dtool

Running the example notebooks in the code repository also requires Jupyter:

.. code-block:: bash

    pip install jupyter

Finally, if you want to run the test suite in the code repository, you'll need
pytest.

.. code-block:: bash
    
    pip install pytest

Installing with ``conda``
~~~~~~~~~~~~~~~~~~~~~~~~~

You can install dtoolAI with conda as follows:

.. code-block:: bash

    conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
    conda install dtoolcore dtool-http dtoolai -c dtool 

This first installs a version of Pytorch known to work with dtoolAI. If you would
like to install the whole ``dtool`` command line suite, you'll need to use pip:

.. code-block:: bash

    pip install dtool