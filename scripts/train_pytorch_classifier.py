import click

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dtoolpytorch import DtoolDataSet

from classifier import BigSeedCNN


def train_pytorch_classifier(dataset_uri):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = DtoolDataSet(dataset_uri, 'training')
    valid_ds = DtoolDataSet(dataset_uri, 'validation')

    print("Training set has {} examples.".format(len(train_ds)))
    print("Validation set has {} examples.".format(len(valid_ds)))

    model = BigSeedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_dl = DataLoader(train_ds, batch_size=16)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(train_dl, 0):
            xb, yb = data
            xb = xb.to(device)
            yb = yb.to(device)
            model.to(device)

            optimizer.zero_grad()

            outputs = model(xb)

            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss))
                running_loss = 0.0


@click.command()
@click.argument('dataset_uri')
@click.argument('output_fpath')
def main(dataset_uri, output_fpath):

    model = train_pytorch_classifier(dataset_uri)


if __name__ == "__main__":
    main()
