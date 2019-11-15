
import click
import torch

from torch.utils.data import DataLoader

from cnn import Net
from data import TensorDataSet


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
@click.argument('test_data_uri')
def main(test_data_uri):

    model = Net()
    model.load_state_dict(torch.load("model.pt"))

    test_tds = TensorDataSet(test_data_uri, test=True)
    test_dl = DataLoader(test_tds, batch_size=64, shuffle=True)

    test(model, test_dl, test_tds)


if __name__ == "__main__":
    main()
