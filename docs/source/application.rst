
Using a trained network model
-----------------------------

In this first example, we'll look at how to apply a trained network to an image
that's new to the network. We'll then look at how dtoolAI allows us to find out
information about the data on which the model was trained and how it was
trained.

Applying the network to new data
--------------------------------

TO AN image

.. code-block:: console

    python scripts/apply_model_to_image.py scratch/models/conv2dcaltechu2/ joshua1.jpg

    Classified joshua1.jpg as joshua_tree



ALSO TO A DATASET

Application to a dataset is useful for model evaluation and large scale
classification.


Finding out about the network
-----------------------------

We can also find out about the network and how it was trained.
