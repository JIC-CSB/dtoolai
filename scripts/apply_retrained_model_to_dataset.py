import logging

import click
import torch
import dtoolcore

import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dtoolai.models import GenNet
from dtoolai.data import ImageDataSet
from dtoolai.utils import evaluate_model_verbose


def model_from_uri(model_uri):
    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    ds = dtoolcore.DataSet.from_uri(model_uri)
    idn = dtoolcore.utils.generate_identifier('model.pt')
    state_abspath = ds.item_content_abspath(idn)

    model.load_state_dict(torch.load(state_abspath, map_location='cpu'))

    return model


@click.command()
@click.argument('model_ds_uri')
@click.argument('test_ds_uri')
def main(model_ds_uri, test_ds_uri):

    logging.basicConfig(level=logging.DEBUG)

    ids = ImageDataSet(test_ds_uri, usetype='test')

    model = model_from_uri(model_ds_uri)

    dl = DataLoader(ids, batch_size=1)

    evaluate_model_verbose(model, dl)



if __name__ == "__main__":
    main()
