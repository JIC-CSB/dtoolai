Retraining a model
------------------

Deep learning models are powerful, but can be slow to train. Retraining let us
take a model that has already been trained on a large dataset and provide it
with new training data that update its weights. This can give accurate models
much faster than training from scratch, with less data.

Let's look at how to do this using dtoolAI.  

In this example, we'll take a model called `ResNet <https://arxiv.org/abs/1512.03385>`_,
that's been trained on a large image dataset, and retrain it to classify new
types of images that the network has not seen before.

Part 1: With a preprepared dataset 
""""""""""""""""""""""""""""""""""

In this example, we'll use the CalTech 101 objects dataset. We provide a hosted
version of this dataset in a suitable format. If you have the ``dtool`` client
installed, you can view information about this hosted dataset like this:

.. code-block:: bash

    $ dtool readme show http://bit.ly/3aRvimq

    dataset_name: Caltech 101 images subset
    project: dtoolAI demonstration datasets
    authors:
    - Fei-Fei Li
    - Marco Andreetto
    - Marc 'Aurelio Ranzato
    reference: |
    L. Fei-Fei, R. Fergus and P. Perona. One-Shot learning of object
    categories. IEEE Trans. Pattern Recognition and Machine Intelligence.
    origin: http://www.vision.caltech.edu/Image_Datasets/Caltech101/

This version of the CalTech data contains just two object classes - llamas and
hedgehogs. We'll train a network to be able to distinguish these.

Retraining the model
~~~~~~~~~~~~~~~~~~~~

Since we have data available, we can immediately run the retraining process.
dtoolAI provides a helper script to apply its library functions for retraining a
model and capturing metadata:

.. code-block:: bash

    $ mkdir example
    $ python scripts/retrain_model_from_dataset.py http://bit.ly/3aRvimq example hlama

After some information about the training process, you should see some
information about where the model has been written:

.. code-block:: bash

    Wrote trained model (resnet18pretrained) weights to file://N108176/Users/hartleym/projects/ai/dtoolai-p/example/hlama

Applying the retrained model to new images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's evaluate the model. We can first try evaluation on a held-out part of our
training dataset. This dataset contains metadata labelling some parts of the
dataset as training data and some as evaluation data. Our evaluation script
takes advantage of this labelling to score the model:

.. code-block:: bash

    $ python scripts/evaluate_model_on_image_dataset.py example/hlama http://bit.ly/3aRvimq
    Testing model hlama on dataset caltech101.hedgellamas
    23/25 correct

Now we can test the newly trained model. Try downloading this image:

https://en.wikipedia.org/wiki/File:Igel.JPG

Then we can apply our trained model

.. code-block:: bash

    python scripts/apply_model_to_image.py example/hlama Igel.JPG

Part 2: With raw data
"""""""""""""""""""""

We saw above how we could retrain a model using data that's already been
packaged into a dataset. Now let's look at how we can work with raw data, by
first packaging it then applying the same process.

Gathering data
~~~~~~~~~~~~~~

You can use any collection of images. For this example, we'll again use the
Caltech 101 objects dataset. which is available `here <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_.

Download the dataset somewhere accessible and unpack it.

Converting the data into a DataSet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dtoolAI provides a helper script to convert a set of named directories
containing images into a dataset suitable for training a deep learning model.

To use this script, we first need to set up our images in the right layout. The
script requires images to be in subdirectories, each of which is named for the
category it represents, e.g.:

.. code-block:: bash

    new_training_example/
    ├── category1
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── category2
    │   ├── image1.jpg
    │   └── image2.jpg
    └── category3
        ├── image1.jpg
        └── image2.jpg

We can then use  helper script provided by dtoolAI, ``create-image-dataset-from-dirtree`` to
turn this directoy into a training dataset.

Assuming that the images are in a directory called ``new_training_example``, and that the
directory ``example`` exists and that we can write to this directory, we run:

.. code-block:: bash

    create-image-dataset-from-dirtree new_training_example example retraining.input.dataset

or, under Windows:

    create-image-dataset-from-dirtree.exe

This will create a new dataset and report its created URI:

.. code-block:: bash

    Created image dataset at file:///C:/Users/myuser/projects/dtoolai/example/retraining.input.dataset

In this example, we're creating the dataset on local disk, so we would need to copy it to persistent
world accessible storage (such as Amazon S3 or Azure storage) when we publish a DL model based on this
dataset. If you have S3 or Azure credentials set up, you can create persistent datasets directly using
the script described above, changing the ``example`` directory to a base URI as described in the
`dtool documentation <https://dtool.readthedocs.io>`_.

Retraining on the new dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we've created our training dataset, we can run the same training script that we used above
on our new dataset, e.g.:

.. code-block:: bash

    python scripts/retrain_model_from_dataset.py file:///C:/Users/myuser/projects/dtoolai/example/retraining.input.dataset example new.model 