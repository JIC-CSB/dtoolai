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

This tells us some information about what the data are, who created it, and
where we can go to find out more.

Training a network
~~~~~~~~~~~~~~~~~~

Given that

.. code-block:: bash

    python scripts/train_cnn_classifier_from_tensor_dataset.py scratch/datasets/mnist.train scratch/ wipeme

This will produce information about the training process, and then report where
the trained model weights have been written, e.g.:

.. code-block:: bash

    Wrote trained model (simpleScalingCNN) weights to file://N108176/Users/hartleym/projects/ai/dtoolai-p/scratch/wipeme

Applying the trained model to new data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Viewing the trained model metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

