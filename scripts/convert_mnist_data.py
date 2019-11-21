import gzip
import pickle

import click
import dtoolcore

import numpy as np

from sklearn.datasets import fetch_mldata

from dtool_utils.quick_dataset import QuickDataSet


def get_mnist_from_sklearn():
    mnist = fetch_mldata("MNIST original")

    with QuickDataSet('scratch', 'mnist') as qds:
        numpy_fpath = qds.staging_fpath('mnist.npy')
        np.save(numpy_fpath, mnist.data)
        labels_fpath = qds.staging_fpath('labels.npy')
        np.save(labels_fpath, mnist.target)

    tensor_idn = dtoolcore.utils.generate_identifier('mnist.npy')
    qds.put_annotation("tensor_file_idn", tensor_idn)

    image_dim = [1, 28, 28]
    qds.put_annotation("image_dimensions", image_dim)


def main():
    
    get_mnist_from_sklearn()


if __name__ == "__main__":
    main()
