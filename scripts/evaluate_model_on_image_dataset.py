import click
import torch
import dtoolcore

from torch.utils.data import DataLoader

from dtoolai.data import ImageDataSet
from dtoolai.trained import TrainedTorchModel


def evaluate_model(model, dl_eval):

    model.eval()
    correct = 0
    n_labels = 0
    with torch.no_grad():
        for data, label in dl_eval:
            n_labels += len(label)
            Y_pred = model(data)
            pred = Y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    return correct


@click.command()
@click.argument('model_ds_uri')
@click.argument('image_ds_uri')
def main(model_ds_uri, image_ds_uri):

    ids = ImageDataSet(image_ds_uri, usetype='test')
    model = TrainedTorchModel(model_ds_uri)

    dl = DataLoader(ids, batch_size=4)

    print(f"Testing model {model.name} on dataset {ids.name}")

    model.model.eval()

    correct = 0
    with torch.no_grad():
        for data, label in dl:
            Y_pred = model.model(data)
            pred = Y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()


    print(f"{correct}/{len(ids)} correct")


if __name__ == "__main__":
    main()
