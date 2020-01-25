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

Gathering data
~~~~~~~~~~~~~~

First, we'll need some data. In this example, we'll use the CalTech 101 objects
dataset, available `here <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_.

Download the dataset somewhere accessible.

Converting the data into a DataSet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dtoolAI provides a helper script to convert a set of named directories
containing images into a the form of dataset suitable for training a deep
learning model.

To use this script, we first need to 

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


Retraining the model
~~~~~~~~~~~~~~~~~~~~

Now we can run the retraining process. 

* The input dataset
* 

Applying the retrained model to new images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can test the newly trained model. Try downloading this image:

.. code-block:: bash

    python scripts/apply_model_to_image.py scratch/models/conv2dcaltechu2/ joshua1.jpg
