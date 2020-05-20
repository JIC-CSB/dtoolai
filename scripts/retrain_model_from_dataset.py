import click

import torch

from dtoolcore import DerivedDataSetCreator

from dtoolai.data import ImageDataSet
from dtoolai.parameters import Parameters
from dtoolai.models import ResNet18Pretrained
from dtoolai.training import train_model_with_metadata_capture


def train_cnn_from_image_dataset(ids_train, output_base_uri, output_name, params):
    
    n_categories = len(ids_train.cat_encoding)
    params['init_params'] = dict(n_outputs=n_categories)
    model = ResNet18Pretrained(**params.init_params)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    
    with DerivedDataSetCreator(output_name, output_base_uri, ids_train) as output_ds:
        train_model_with_metadata_capture(model, ids_train, optimiser, loss_fn, params, output_ds)

    print(f"Wrote trained model ({model.model_name}) weights to {output_ds.uri}")


@click.command()
@click.argument('input_dataset_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
@click.option('--params')
def main(input_dataset_uri, output_base_uri, output_name, params):
    """Train a classifier from a tensor DataSet.

    \b
    Required arguments:
    input_dataset_uri: The URI of the data that will be used to train the model.
    output_base_uri:   The base URI at which the output model DataSet will be created.
                       Local file URIS (file:///path/to/file) can be specified as absolute
                       or relative filesystem paths, which will be converted to full URIs.
    output_name:       The name for the output DataSet.  

    \b
    Optional arguments:
    params:            A comma separated list of parameters that will override defaults for
                       training the model.
    """

    input_ds_train = ImageDataSet(input_dataset_uri)

    model_params = Parameters(
        batch_size=4,
        learning_rate=0.001,
        n_epochs=1
    )
    if params: model_params.update_from_comma_separated_string(params)

    train_cnn_from_image_dataset(
        input_ds_train,
        output_base_uri,
        output_name,
        model_params
    )


if __name__ == "__main__":
    main()
