Extending dtoolAI
-----------------

dtoolAI provides everything needed to train image classification networks
"out of the box". Different types of Deep Learning network will require both
new models and possibly classes for training data.

New forms of training data
~~~~~~~~~~~~~~~~~~~~~~~~~~

dtoolAI provides two classes for managing training data - ``TensorDataSet`` and
``ImageDataSet``. Our examples use these to train models and capture provenance.

The class should:

* Inherit from ``dtoolai.data.WrappedDataSet``. This ensures that it provides
  both the methods required by Pytorch (to feed into the model) and dtoolAI (to
  capture metadata).
* Implement ``__len__`` which should return how many items are in the dataset.
* Implement ``__getitem__``, which should return either ``torch.Tensor`` objects
  or numpy arrays that Pytorch is capable of converting to tensors.

Instances of this class can then be passed to
``dtoolai.training.train_model_with_metadata_capture``.
