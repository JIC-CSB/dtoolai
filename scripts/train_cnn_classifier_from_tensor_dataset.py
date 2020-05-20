import json

import click

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dtoolcore import DerivedDataSetCreator

from dtoolai.data import TensorDataSet
from dtoolai.models import GenNet
from dtoolai.training import train_model_with_metadata_capture
from dtoolai.parameters import Parameters


def train_cnn_from_tensor_dataset(tds_train, output_base_uri, output_name, params):

    params['input_dim'] = tds_train.dim
    params['input_channels'] = tds_train.input_channels
    params['init_params'] = dict(input_channels=tds_train.input_channels, input_dim=tds_train.dim)
    model = GenNet(**params['init_params'])
    loss_fn = torch.nn.NLLLoss()
    optimiser = optim.SGD(model.parameters(), lr=params.learning_rate)

    with DerivedDataSetCreator(output_name, output_base_uri, tds_train) as output_ds:
        train_model_with_metadata_capture(model, tds_train, optimiser, loss_fn, params, output_ds)

    print(f"Wrote trained model ({model.model_name}) weights to {output_ds.uri}")


@click.command()
@click.argument('train_dataset_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
@click.option('--params')
@click.option('--test-dataset-uri')
def main(train_dataset_uri, output_base_uri, output_name, params, test_dataset_uri):
    """Train a classifier from a tensor DataSet.

    \b
    Required arguments:
    train_dataset_uri: The URI of the data that will be used to train the model.
    output_base_uri:   The base URI at which the output model DataSet will be created.
                       Local file URIS (file:///path/to/file) can be specified as absolute
                       or relative filesystem paths, which will be converted to full URIs.
    output_name:       The name for the output DataSet.  

    \b
    Optional arguments:
    params:            A comma separated list of parameters that will override defaults for
                       training the model.
    test_dataset_uri   A URI with a test DataSet that will be used to evaluate the model.
                       during and after training. As for output_base_uri, filesyste, paths
                       can be used and will be converted to file URIS.  
    """

    input_ds_train = TensorDataSet(train_dataset_uri)   

    model_params = Parameters(
        batch_size=128,
        learning_rate=0.01,
        n_epochs=1
    )
    if params:
        model_params.update_from_comma_separated_string(params)

    train_cnn_from_tensor_dataset(
        input_ds_train,
        output_base_uri,
        output_name,
        model_params
    )


if __name__ == "__main__":
    main()
