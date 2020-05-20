import click

from dtoolcore import DerivedDataSetCreator

from torch.utils.data import DataLoader

import torch.optim as optim

import torch.nn.functional as F

from dtoolai.data import ImageDataSet
from dtoolai.models import GenNet
from dtoolai.utils import train
from dtoolai.parameters import Parameters


def train_cnn_from_image_dataset(ids, output_base_uri, output_name, params):

    dl = DataLoader(ids, batch_size=params.batch_size, shuffle=True)

    model = GenNet(ids.input_channels, ids.dim)
    loss_fn = F.nll_loss
    optimiser = optim.SGD(model.parameters(), lr=params.learning_rate)

    with DerivedDataSetCreator(output_name, output_base_uri, ids) as output_ds:
        train(model, dl, optimiser, loss_fn, params.n_epochs)
        output_ds.readme_dict['parameters'] = params.parameter_dict
        output_ds.readme_dict['model_name'] = model.model_name


@click.command()
@click.argument('input_dataset_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(input_dataset_uri, output_base_uri, output_name):
    """Train a classifier from an image DataSet.

    \b
    Arguments:
    input_dataset_uri: The URI of the data that will be used to train the model.
    output_base_uri:   The base URI at which the output model DataSet will be created.
                       Local file URIS (file:///path/to/file) can be specified as absolute
                       or relative filesystem paths, which will be converted to full URIs.
    output_name:       The name for the output DataSet.  
    """

    input_ds = ImageDataSet(input_dataset_uri)

    params = Parameters(
        batch_size=4,
        learning_rate=0.01,
        n_epochs=20
    )

    train_cnn_from_image_dataset(input_ds, output_base_uri, output_name, params)


if __name__ == "__main__":
    main()
