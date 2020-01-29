from collections import Counter

import click

from dtoolai.data import TensorDataSet


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    tds = TensorDataSet(dataset_uri)

    print(f"Dataset has {len(tds)} items")

    print(Counter(tds.labels))


if __name__ == "__main__":
    main()
