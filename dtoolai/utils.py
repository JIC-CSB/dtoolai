import os
import time
import pprint
from collections import defaultdict
from pathlib import Path

import click
import torch
import dtoolcore
import numpy as np

from dtool_utils.quick_dataset import QuickDataSet

from dtoolai.trained import TrainedTorchModel


IMAGEDS_README_TEMPLATE = """---
Created from directory: {dirpath}
Created by: dtoolAI.utils:image_dataset_from_dirtree
"""


@click.command()
@click.argument('dirtree_dirpath')
@click.argument('output_base_uri')
@click.argument('output_name')
def image_dataset_from_dirtree(dirtree_dirpath, output_base_uri, output_name):

    categories = [d for d in os.listdir(dirtree_dirpath)]

    def relpath_from_srcpath(srcpath, cat):
        return cat + '/' + os.path.basename(srcpath)

    items_to_include = {}
    for cat in categories:
        srcpath_iter = (Path(dirtree_dirpath) / cat).iterdir()
        items_to_include.update({
            srcpath: (relpath_from_srcpath(srcpath, cat), cat)
            for srcpath in srcpath_iter
        })

    category_encoding = {c: n for n, c in enumerate(categories)}
    with QuickDataSet(output_base_uri, output_name) as output_ds:
        for srcpath, (relpath, cat) in items_to_include.items():
            handle = output_ds.put_item(srcpath, relpath)
            output_ds.add_item_metadata(handle, 'category', cat)
        output_ds.put_annotation('category_encoding', category_encoding)
        output_ds.put_annotation("dtoolAI.inputtype", "ImageDataSet")
        abs_dirpath = os.path.abspath(dirtree_dirpath)
        output_ds.put_readme(IMAGEDS_README_TEMPLATE.format(dirpath=abs_dirpath))

    return output_ds.uri


def coerce_to_target_dim(im, input_format):
    """Convert a PIL image to a fixed size and number of channels."""

    mode_map = {
        1: 'L',
        3: 'RGB',
        4: 'RGBA'
    }

    if im.mode not in mode_map.values():
        raise Exception(f"Unknown image mode: {im.mode}")

    ch, tdimw, tdimh = input_format
    if ch not in mode_map:
        raise Exception(f"Unsupported input format: {input_format}")

    cdimw, cdimh = im.size
    if (cdimw, cdimh) != (tdimw, tdimh):
        im = im.resize((tdimw, tdimh))

    if mode_map[ch] != im.mode:
        im = im.convert(mode_map[ch])

    return im


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

    return correct


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
