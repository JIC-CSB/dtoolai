import time
import pprint
from collections import defaultdict

import click
import torch
import dtoolcore
import numpy as np

from dtoolai.trained import TrainedTorchModel


@click.command()
@click.argument('model_uri')
def print_provenance(model_uri):
    ttm = TrainedTorchModel(model_uri)

    print(f"Network architecture name: {ttm.model_name}")

    model_parameters = ttm.dataset.get_annotation("model_parameters")

    print(f"Model training parameters: {pprint.pformat(model_parameters)}")

    source_ds_uri = ttm.dataset.get_annotation("source_ds_uri")
    print(f"Source dataset URI: {source_ds_uri}")

    source_ds = dtoolcore.DataSet.from_uri(source_ds_uri)
    print(f"Source dataset name: {source_ds.name}")

    print(f"Source dataset readme:\n{source_ds.get_readme_content()}")


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

        print(f"{correct}/{n_labels}")


def evaluate_model_verbose(model, dl_eval):

    model.eval()
    correct = 0
    n_labels = 0
    with torch.no_grad():
        for data, label in dl_eval:

            model_input = data
            print(model_input)
            model_input_np = model_input.squeeze().cpu().numpy()
            model_input_t = np.transpose(model_input_np, (1, 2, 0))
            imsave('scratch/model_input_t.png', model_input_t)

            print(model_input.shape, model_input.min(), model_input.max(), model_input.dtype)
            n_labels += len(label)
            Y_pred = model(data)
            print(Y_pred)
            pred = Y_pred.argmax(dim=1, keepdim=True)
            print(pred, label)
            correct += pred.eq(label.view_as(pred)).sum().item()

        print(f"{correct}/{n_labels}")


def train(model, dl, optimiser, loss_fn, n_epochs, dl_eval=None):
    """

    Returns: dictionary containing training history.
    """
    
    history = defaultdict(list)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(n_epochs):
        epoch_loss = 0
        t_start = time.time()
        model.train()
        for n, (data, label) in enumerate(dl):
            data = data.to(device)
            label = label.to(device)
            optimiser.zero_grad()
            Y_pred = model(data)

            loss = loss_fn(Y_pred, label)
            loss.backward()
            epoch_loss += loss.item()

            optimiser.step()

            if n % 10 == 0:
                print(f"  Epoch {epoch}, batch {n}/{len(dl)}, running loss {epoch_loss}")
        history["epoch_loss"].append(epoch_loss)
        print(f"Epoch {epoch}, training loss {epoch_loss}, time {time.time()-t_start}")
        if dl_eval:
            model.eval()
            valid_loss = 0
            for n, (data, label) in enumerate(dl):
                data = data.to(device)
                label = label.to(device)
                Y_pred = model(data)
                valid_loss += loss_fn(Y_pred, label).item()
            print(f"Validation loss {valid_loss}")
            history["valid_loss"].append(valid_loss)
            # evaluate_model(model, dl_eval)

    return history
