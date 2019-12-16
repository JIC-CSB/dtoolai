import click
import torch
import torch.utils.data
import dtoolcore

from torch.utils.data import DataLoader

import torch.optim as optim

import torch.nn.functional as F

import numpy as np

from dtoolai.models import GenNet
from dtoolai.data import TensorDataSet

from dtool_utils.derived_dataset import DerivedDataSet


def train(model, dl, optimiser, n_epochs):
    total_loss = 0
    model.train()
    for n, (data, label) in enumerate(dl):
        optimiser.zero_grad()
        Y_pred = model(data)

        # print(data.shape, Y_pred.shape, label.shape)
        loss = F.nll_loss(Y_pred, label)
        loss.backward()
        total_loss += loss.item()

        optimiser.step()

        if n % 10 == 0:
            print(f"Batch {n}/{len(dl)}, running loss {total_loss}")

    print("Epoch training loss", total_loss)


def test(model, dl, tds):

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in dl:
                Y_pred = model(data)
                test_loss += F.nll_loss(Y_pred, label).item()
                pred = Y_pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        print("Test loss:", test_loss)
        print(f"{correct}/{len(tds)} correct")


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    tds = TensorDataSet(dataset_uri)
    test_tds = TensorDataSet(dataset_uri, test=True)

    dl = DataLoader(tds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_tds, batch_size=64, shuffle=True)

    model = GenNet(tds.input_channels, tds.dim)

    optimiser = optim.SGD(model.parameters(), lr=0.01)

    test(model, test_dl, test_tds)

    output_base_uri = "scratch/models"
    output_name = "conv2dmnist"
    with DerivedDataSet(output_base_uri, output_name, tds, overwrite=True) as output_ds:
    # TODO - move train/val and epochs into the training loop
        for epoch in range(1):
            train(model, dl, optimiser, 2)
            test(model, test_dl, test_tds)

        model_output_fpath = output_ds.staging_fpath("model.pt")
        torch.save(model.state_dict(), model_output_fpath)




if __name__ == "__main__":
    main()
