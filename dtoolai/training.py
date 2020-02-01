import json

import torch
from torch.utils.data import DataLoader

from dtoolai.utils import (
    train,
    nested_dict_as_string,
    readme_dict_from_source_dataset
)


def train_model_with_metadata_capture(model, ds_train, optimiser, loss_fn, params, output_ds, ds_valid=None):
    """Train a Pytorch model from a dtoolAI dataset, capturing training data
    metadata and training parameters during the process.

    Args:
        model (torch.nn): A Pytorch model to be trained.
        ds_train: A dtoolAI/Pytorch compatible dataset object. See below for
            details.
        optimiser: The Pytorch optimiser to be used during training.
        loss_fn: The loss function that should be applied to the model output.
        params: A dtoolai.parameters.Parameters class holding training
            parameters.
        output_ds: The output dataset creator to which the trained model's
            weights and parameters will be written.

    This function will train the model on the data it is given, using the
    supplied loss function, optimiser and parameters. It will ensure that the
    parameters and training data provenance are captured and stored with the
    model.

    The dataset parameter (ds_train) must provide __getitem__ and __len__.
    """
    dl_train = DataLoader(ds_train, batch_size=params.batch_size, shuffle=True)

    params['input_dim'] = ds_train.dim
    params['input_channels'] = ds_train.input_channels
    params['optimiser_name'] = optimiser.__class__.__name__
    params['loss_func'] = loss_fn.__class__.__name__

    history = train(model, dl_train, optimiser, loss_fn, params.n_epochs)

    model_output_fpath = output_ds.prepare_staging_abspath_promise('model.pt')
    torch.save(model.state_dict(), model_output_fpath)
    history_output_fpath = output_ds.prepare_staging_abspath_promise('history.json')
    with open(history_output_fpath, 'w') as fh:
        json.dump(history, fh)

    output_ds.put_annotation("model_parameters", params.parameter_dict)
    output_ds.put_annotation("model_name", f"dtoolai.{model.model_name}")

    try:
        has_categorical_output = model.categorical_output
    except AttributeError:
        has_categorical_output = False
    if has_categorical_output:
        cat_encoding = ds_train.get_annotation("category_encoding")
        output_ds.put_annotation("category_encoding", cat_encoding)

    readme_dict = readme_dict_from_source_dataset(ds_train)
    readme_dict['parameters'] = params.parameter_dict
    readme_dict['model_name'] = model.model_name
    
    dict_repr = nested_dict_as_string(readme_dict)
    output_ds.put_readme(dict_repr)
