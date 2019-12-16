import click
import torch
import dtoolcore

import torch.nn.functional as F
from torch.utils.data import DataLoader

from dtoolai.models import GenNet
from dtoolai.data import TensorDataSet
from dtoolai.trained import TrainedTorchModel

def model_from_uri(model_uri):
    model = GenNet(3, 32)

    ds = dtoolcore.DataSet.from_uri(model_uri)
    idn = dtoolcore.utils.generate_identifier('model.pt')
    state_abspath = ds.item_content_abspath(idn)

    model.load_state_dict(torch.load(state_abspath, map_location='cpu'))

    return model


@click.command()
@click.argument('model_ds_uri')
@click.argument('test_ds_uri')
def main(model_ds_uri, test_ds_uri):

    tds = TensorDataSet(test_ds_uri, test=True)
    model = TrainedTorchModel(model_ds_uri)

    # model = model_from_uri(model_ds_uri)

    dl = DataLoader(tds, batch_size=128)

    model.model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in dl:
            Y_pred = model.model(data)
            test_loss += F.nll_loss(Y_pred, label).item()
            pred = Y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    # print("Test loss:", test_loss)
    print(f"{correct}/{len(tds)} correct")


if __name__ == "__main__":
    main()
