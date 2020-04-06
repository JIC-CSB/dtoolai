import logging

import click

import torch

from dtoolcore import DerivedDataSetCreator

from dtoolai.data import TensorDataSet
from dtoolai.parameters import Parameters
from dtoolai.models import GenNet
from dtoolai.training import train_model_with_metadata_capture
from dtoolai.utils import BaseCallBack


class ExampleCallback(BaseCallBack):

    def __init__(self):
        pass

    def on_epoch_end(self, epoch, model, history):
        print("Hello!")


def train_cnn_from_tensor_dataset(tds_train, output_base_uri, output_name, params, callbacks):

    params['input_dim'] = tds_train.dim
    params['input_channels'] = tds_train.input_channels
    params['init_params'] = dict(
        input_channels=tds_train.input_channels, input_dim=tds_train.dim)
    model = GenNet(**params['init_params'])
    loss_fn = torch.nn.NLLLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=params.learning_rate)

    with DerivedDataSetCreator(output_name, output_base_uri, tds_train) as output_ds:
        train_model_with_metadata_capture(
            model, tds_train, optimiser, loss_fn, params, output_ds, callbacks=callbacks
        )

    print(
        f"Wrote trained model ({model.model_name}) weights to {output_ds.uri}"
    )



@click.command()
def main():
    logging.basicConfig(level=logging.INFO)
    model_params = Parameters(
        batch_size=128,
        learning_rate=0.01,
        n_epochs=2
    )

    mnist_train_uri = "http://bit.ly/2uqXxrk"
    train_ds = TensorDataSet(mnist_train_uri)

    example_callback = ExampleCallback()
    train_cnn_from_tensor_dataset(
        train_ds, "scratch", "test", model_params, [example_callback]
    )


if __name__ == "__main__":
    main()
