import click
import torch
import torch.utils.data
import dtoolcore

from torch.utils.data import DataLoader

import torch.optim as optim

import torch.nn.functional as F

import numpy as np

from cnn import Net

from sklearn.model_selection import train_test_split


class WrappedDataSet(torch.utils.data.Dataset):

    def __init__(self, uri):
        self.dataset = dtoolcore.DataSet.from_uri(uri)

    @property
    def name(self):
        return self.dataset.name

    @property
    def uri(self):
        return self.dataset.uri

    @property
    def uuid(self):
        return self.dataset.uuid


# TODO - FIX THE OBJECT TYPE 
class TensorDataSet(WrappedDataSet):

    def __init__(self, uri, test=False):
        super().__init__(uri)

        idn = dtoolcore.utils.generate_identifier("mnist.npy")
        npy_fpath = self.dataset.item_content_abspath(idn)
        self.X = np.load(npy_fpath, mmap_mode=None)

        labels_idn = dtoolcore.utils.generate_identifier("labels.npy")
        labels_fpath = self.dataset.item_content_abspath(labels_idn)
        self.y = np.load(labels_fpath, mmap_mode=None)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=0)

        if test:
            self.tensor = self.X_test
            self.labels = self.y_test
        else:
            self.tensor = self.X_train
            self.labels = self.y_train

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, index):
        raw = self.tensor[index]
        scaledfloat = raw.astype(np.float32) / 255

        label = self.labels[index]

        return scaledfloat.reshape(1, 28, 28), int(label)


def train(model, dl, optimiser, n_epochs):
    total_loss = 0
    model.train()
    for n, (data, label) in enumerate(dl):
        optimiser.zero_grad()
        Y_pred = model(data)

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


    model = Net()

    optimiser = optim.SGD(model.parameters(), lr=0.01)

    test(model, test_dl, test_tds)

    for epoch in range(2):
        train(model, dl, optimiser, 2)
        test(model, test_dl, test_tds)





if __name__ == "__main__":
    main()
