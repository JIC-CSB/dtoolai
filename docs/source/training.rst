Training a new model
--------------------

In this example we'll look at one of the "hello world" example problems of
training deep learning networks - handwritten digit recognition. We'll use the
MNIST dataset, consisting of 70,000 labelled handwritten digits between 0 and
9 to train a convolutional neural network.

The dataset
~~~~~~~~~~~

In this case, we've created a dtool DataSet from the MNIST data. We can use the
dtool CLI to see what we know about this DataSet:

.. code-block:: bash

    $ dtool readme show scratch/datasets/mnist.train
    ---
    dataset_name: MNIST handwritten digits
    project: dtoolAI demonstration datasets
    authors:
    - Yann LeCun
    - Corinna Cortes
    - Christopher J.C. Burges
    origin: http://yann.lecun.com/exdb/mnist/
    usetype: train

This tells us some information about what the data are, who created them, and
where we can go to find out more.

Training a network
~~~~~~~~~~~~~~~~~~

We'll start by using one of the helper scripts from dtoolAI to train a CNN.
Later, we'll look at what the script is doing.

.. code-block:: bash

    python scripts/train_cnn_classifier_from_tensor_dataset.py scratch/datasets/mnist.train scratch/ wipeme

This will produce information about the training process, and then report where
the trained model weights have been written, e.g.:

.. code-block:: bash

    Wrote trained model (simpleScalingCNN) weights to file://N108176/Users/hartleym/projects/ai/dtoolai-p/scratch/wipeme

Applying the trained model to test data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to test our model is on another preprepared dataset - the 

We have test data available

Viewing the trained model metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's look at how 

.. code-block:: bash

    dtoolai provenance


What the code is doing
~~~~~~~~~~~~~~~~~~~~~~

Let's dig into what the library code is doing. We'll work through the MNIST
example


