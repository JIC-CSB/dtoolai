import click

from data import TensorDataSet

from torch.utils.data import DataLoader


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    tds = TensorDataSet(dataset_uri)
    dl = DataLoader(tds, batch_size=64)

    tensor, label = next(iter(dl))

    print(tensor.shape)
    print(label.shape)


if __name__ == "__main__":
    main()
