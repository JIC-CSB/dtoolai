import gzip
import pickle

import click
import dtoolcore

import numpy as np

from sklearn.datasets import fetch_mldata

from dtoolai.data import create_tensor_dataset_from_arrays


README_TEMPLATE = """---
dataset_name: MNIST handwritten digits
project: dtoolAI demonstration datasets
authors:
  - Yann LeCun
  - Corinna Cortes
  - Christopher J.C. Burges
origin: http://yann.lecun.com/exdb/mnist/
usetype: {usetype}
"""


def get_mnist_from_sklearn():
    mnist = fetch_mldata("MNIST original")

    return mnist


def create_dataset_from_mnist_data(mnist, output_base_uri, output_prefix):

    output_name_train = output_prefix + '.train'
    output_name_test = output_prefix + '.test'

    image_dim = [1, 28, 28]
    n_train = 60000
    
    train_uri = create_tensor_dataset_from_arrays(
        output_base_uri,
        output_name_train,
        mnist.data[:n_train],
        mnist.target[:n_train],
        image_dim,
        README_TEMPLATE.format(usetype='train')
    )
    cat_encoding = {n: n for n in range(10)}
    ds = dtoolcore.DataSet.from_uri(train_uri)
    ds.put_annotation("category_encoding", cat_encoding)

    test_uri = create_tensor_dataset_from_arrays(
        output_base_uri,
        output_name_test,
        mnist.data[n_train:],
        mnist.target[n_train:],
        image_dim,
        README_TEMPLATE.format(usetype='test')
    )
    ds = dtoolcore.DataSet.from_uri(test_uri)
    ds.put_annotation("category_encoding", cat_encoding)


@click.command()
@click.argument('output_base_uri')
@click.argument('output_prefix')
def main(output_base_uri, output_prefix):
    
    mnist = get_mnist_from_sklearn()
    create_dataset_from_mnist_data(mnist, output_base_uri, output_prefix)


if __name__ == "__main__":
    main()
