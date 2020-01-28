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

    $ dtool readme show http://bit.ly/2uqXxrk
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

    mkdir example
    python scripts/train_cnn_classifier_from_tensor_dataset.py http://bit.ly/2uqXxrk example mnistcnn

This will produce information about the training process, and then report where
the trained model weights have been written, e.g.:

.. code-block:: bash

    Wrote trained model (simpleScalingCNN) weights to file://N108176/Users/hartleym/projects/ai/dtoolai-p/example/mnistcnn

Applying the trained model to test data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to test our model is on another preprepared dataset - this
allows us to quickly apply the model to many ready-labelled images and calculate
its accuracy.

We have provided the MNIST test data as a separate dtool DataSet for this
purpose, and we can apply our new model to this dataset like this:

.. code-block:: bash

    $ python scripts/apply_model_to_tensor_dataset.py \
        example/mnistcnn http://bit.ly/2NVFGQd
    7929/10000 correct

If we want to improve the model's accuracy, we could try training it for longer.
For example, to train it for 5 epochs (loops through the training dataset)
rather than one, we can run our script again:

    python scripts/train_cnn_classifier_from_tensor_dataset.py \ 
        http://bit.ly/2uqXxrk example mnistcnn --params n_epochs=5

This will train the model for longer.

Viewing the trained model metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the core features of dtoolAI is capture of references to training data
and metadata about the training process. Let's look at how we access those
captured data for our newly trained model.

dtoolai provides a helper script, ``dtoolai-provenance`` for this purpose. This
will show a model's training metadata, the references to its training data, then
the metadata for those training data.

.. code-block:: bash

    $ dtoolai-provenance example/mnistcnn/

    Network architecture name: dtoolai.simpleScalingCNN
    Model training parameters: {'batch_size': 128,
    'init_params': {'input_channels': 1, 'input_dim': 28},
    'input_channels': 1,
    'input_dim': 28,
    'learning_rate': 0.01,
    'n_epochs': 1,
    'optimiser_name': 'SGD'}
    Source dataset URI: http://bit.ly/2uqXxrk
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

We can see that the model dataset contains both information about how the model
was trained (learning_rate, n_epochs and so on) as well as the reference to the
training data, which we can follow to show its provenance.

What the code is doing
~~~~~~~~~~~~~~~~~~~~~~

We provide the Jupyter notebook TrainingExplained.ipynb to show how the training
script uses dtoolAI's library functions and classes to make capturing training
metadata and parameters easier.



