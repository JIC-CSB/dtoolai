import click

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dtool_utils.derived_dataset import DerivedDataSet

from dtoolai.data import TensorDataSet
from dtoolai.models import GenNet
from dtoolai.utils import train
from dtoolai.parameters import Parameters


def train_cnn_from_tensor_dataset(tds_train, output_base_uri, output_name, params):

    dl_train = DataLoader(tds_train, batch_size=params.batch_size, shuffle=True)

    params['input_dim'] = tds_train.dim
    params['input_channels'] = tds_train.input_channels
    params['init_params'] = dict(input_channels=tds_train.input_channels, input_dim=tds_train.dim)
    model = GenNet(**params['init_params'])
    loss_fn = F.nll_loss
    optimiser = optim.SGD(model.parameters(), lr=params.learning_rate)

    # FIXME
    params['optimiser_name'] = optimiser.__class__.__name__

    # FIXME - probably should be in tensor dataset
    cat_encoding = {n: n for n in range(10)}

    with DerivedDataSet(output_base_uri, output_name, tds_train, overwrite=True) as output_ds:
        train(model, dl_train, optimiser, loss_fn, params.n_epochs)
        output_ds.readme_dict['parameters'] = params.parameter_dict
        output_ds.readme_dict['model_name'] = model.model_name
        model_output_fpath = output_ds.staging_fpath('model.pt')
        torch.save(model.state_dict(), model_output_fpath)

        output_ds.put_annotation("category_encoding", cat_encoding)
        output_ds.put_annotation("model_parameters", params.parameter_dict)
        output_ds.put_annotation("model_name", f"dtoolai.{model.model_name}")

    print(f"Wrote trained model ({model.model_name}) weights to {output_ds.uri}")


@click.command()
@click.argument('train_dataset_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
@click.option('--params')
@click.option('--test-dataset-uri')
def main(train_dataset_uri, output_base_uri, output_name, params, test_dataset_uri):

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
