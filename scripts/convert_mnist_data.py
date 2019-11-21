import gzip
import pickle

import click
import dtoolcore

import numpy as np

from sklearn.datasets import fetch_mldata

from dtool_utils.quick_dataset import QuickDataSet


README_TEMPLATE = """---
dataset_name: MNIST handwritten digits
project: dtoolAI demonstration datasets
authors:
  - Yann LeCun
  - Corinna Cortes
  - Christopher J.C. Burges
origin: http://yann.lecun.com/exdb/mnist/
"""


def get_mnist_from_sklearn():
    mnist = fetch_mldata("MNIST original")

    return mnist


def create_dataset_from_mnist_data(mnist, output_base_uri, output_name):

    with QuickDataSet(output_base_uri, output_name) as qds:
        numpy_fpath = qds.staging_fpath('mnist.npy')
        np.save(numpy_fpath, mnist.data)
        labels_fpath = qds.staging_fpath('labels.npy')
        np.save(labels_fpath, mnist.target)

    tensor_idn = dtoolcore.utils.generate_identifier('mnist.npy')
    qds.put_annotation("tensor_file_idn", tensor_idn)

    image_dim = [1, 28, 28]
    qds.put_annotation("image_dimensions", image_dim)

    qds.put_readme(README_TEMPLATE)


@click.command()
@click.argument('output_base_uri')
@click.argument('output_name')
def main(output_base_uri, output_name):
    
    mnist = get_mnist_from_sklearn()
    create_dataset_from_mnist_data(mnist, output_base_uri, output_name)


if __name__ == "__main__":
    main()
