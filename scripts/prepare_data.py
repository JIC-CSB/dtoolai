import gzip
import pickle

import click

import numpy as np

from sklearn.datasets import fetch_mldata

# @click.command()
# @click.argument('pickle_gz_fpath')
# def main(pickle_gz_fpath):

#     with gzip.open(pickle_gz_fpath, 'rb') as fh:
#         print(pickle.load(fh))

from imageio import imsave


from dtool_utils.quick_dataset import QuickDataSet


def get_mnist_from_sklearn():
    mnist = fetch_mldata("MNIST original")

    with QuickDataSet('scratch', 'mnist') as qds:
        numpy_fpath = qds.staging_fpath('mnist.npy')
        np.save(numpy_fpath, mnist.data)
        labels_fpath = qds.staging_fpath('labels.npy')
        np.save(labels_fpath, mnist.target)


def main():
    
    get_mnist_from_sklearn()

    


if __name__ == "__main__":
    main()
