import click

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
import torch.functional as F
import torch.optim as optim

from dtool_utils.derived_dataset import DerivedDataSet

from dtoolai.data import ImageDataSet
from dtoolai.utils import train, evaluate_model, evaluate_model_verbose
from dtoolai.parameters import Parameters
from dtoolai.models import ResNet18Pretrained


def parse_comma_separated_param_list(raw_string):

    return dict(p.split('=') for p in raw_string.split(','))


@click.command()
@click.argument('input_dataset_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
@click.option('--params')
def main(input_dataset_uri, output_base_uri, output_name, params):

    model_params = Parameters(
        batch_size=4,
        learning_rate=0.001,
        n_epochs=1
    )
    if params: model_params.update_from_comma_separated_string(params)

    ids = ImageDataSet(input_dataset_uri)
    dl = DataLoader(ids, batch_size=model_params.batch_size, shuffle=True)

    n_categories = len(ids.cat_encoding)
    model_params['init_params'] = dict(n_outputs=n_categories)
    model = ResNet18Pretrained(**model_params.init_params)

    ids_test = ImageDataSet(input_dataset_uri, usetype='test')
    dl_test = DataLoader(ids_test, batch_size=model_params.batch_size, shuffle=True)
    evaluate_model(model, dl_test)

    model_params['input_dim'] = ids.dim
    model_params['input_channels'] = ids.input_channels

    criterion = nn.CrossEntropyLoss()
    optimiser_ft = optim.SGD(model.parameters(), lr=model_params.learning_rate, momentum=0.9)
    with DerivedDataSet(output_base_uri, output_name, ids, overwrite=True) as output_ds:
        train(model, dl, optimiser_ft, criterion, model_params.n_epochs, dl_eval=dl_test)
        output_ds.readme_dict['parameters'] = model_params.parameter_dict
        output_ds.readme_dict['model_name'] = model.model_name
        model_output_fpath = output_ds.staging_fpath('model.pt')
        torch.save(model.state_dict(), model_output_fpath)
        output_ds.put_annotation("category_encoding", ids.cat_encoding)
        output_ds.put_annotation("model_parameters", model_params.parameter_dict)
        output_ds.put_annotation("model_name", f"dtoolai.{model.model_name}")

    evaluate_model(model, dl_test)


if __name__ == "__main__":
    main()
