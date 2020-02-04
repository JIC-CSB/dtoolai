Using a trained network model
-----------------------------

In this first example, we'll look at how to apply a trained network to an image
that's new to the network. We'll then look at how dtoolAI allows us to find out
information about the data on which the model was trained and how it was
trained.

Applying the network to new data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's start by trying to classify a new image. Download the following image:

.. image:: non_mnist_three.png

Now run the script ``apply_model_to_image.py`` in the ``scripts/`` directory
of the dtoolAI repository on the image, e.g.:

.. code-block:: console

    $ python scripts/apply_model_to_image.py http://bit.ly/2tbPzSB ~/Downloads/three.png
    Classified /Users/hartleym/Downloads/three.png as 3

We've applied an existing model to a new image.

Finding out about the network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can also find out about the network and how it was trained. For this, we'll
use the command ``dtoolai-provenance`` that's provided when you install the
dtoolAI package. This command displays data about a trained model including the
training data URI. It then attempts to follow that URI to give more information
about the training data:

.. code-block:: console

    $ dtoolai-provenance http://bit.ly/2tbPzSB
    Network architecture name: dtoolai.simpleScalingCNN
    Model training parameters: {'batch_size': 128,
    'init_params': {'input_channels': 1, 'input_dim': 28},
    'input_channels': 1,
    'input_dim': 28,
    'learning_rate': 0.01,
    'loss_func': 'NLLLoss',
    'n_epochs': 10,
    'optimiser_name': 'SGD'}
    Source dataset URI: http://bit.ly/2NVFGQd
    Source dataset name: mnist.train
    Source dataset readme:
    ---
    dataset_name: MNIST handwritten digits
    project: dtoolAI demonstration datasets
    authors:
    - Yann LeCun
    - Corinna Cortes
    - Christopher J.C. Burges
    origin: http://yann.lecun.com/exdb/mnist/
    usetype: train

Here we see that model's network architecture is ``simpleScalingCNN`` from the
dtoolAI package, some more information about the training parameters then, at
the bottom, some information about the training data for the model.

Next, we'll look at how to train a model like this one.
