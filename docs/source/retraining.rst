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

Retraining the model
~~~~~~~~~~~~~~~~~~~~

Now we can run the retraining process. dtoolAI provides a helper script to
apply its library functions for retraining a model and capturing metadata. 

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


Choose two. In this example, we're going to use hedgehogs and llamas.

