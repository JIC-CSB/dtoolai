import pickle

from pathlib import Path

import click
import dtoolcore

import numpy as np

# from imageio import imsave

from dtool_utils.quick_dataset import QuickDataSet


def convert_cifar10_data(cifar_dirpath, output_base_uri, output_name):

    with open(cifar_dirpath/'data_batch_1', 'rb') as fh:
        cifar_dict = pickle.load(fh, encoding='bytes')

    with QuickDataSet(output_base_uri, output_name) as qds:
        data_abspath = qds.staging_fpath('cifar.npy')
        np.save(data_abspath, cifar_dict[b'data'])
        labels_fpath = qds.staging_fpath('labels.npy')
        np.save(labels_fpath, cifar_dict[b'labels'])

    tensor_idn = dtoolcore.utils.generate_identifier('cifar.npy')
    qds.put_annotation("tensor_file_idn", tensor_idn)
    
    image_dim = [3, 32, 32]
    qds.put_annotation("image_dimensions", image_dim)


@click.command()
@click.argument('cifar_base_dirpath')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(cifar_base_dirpath, output_base_uri, output_name):
    
    cifar_dirpath = Path(cifar_base_dirpath)
    convert_cifar10_data(cifar_dirpath, output_base_uri, output_name)
    # cifar_dirpath = Path(cifar_base_dirpath)

    # with open(cifar_dirpath/'data_batch_1', 'rb') as fh:
    #     d = pickle.load(fh, encoding='bytes')

    # data = d[b'data']

    # first = data[1]

    # reshaped = first.reshape((3, 32, 32))
    # transposed = reshaped.transpose((1, 2, 0))

    # print(transposed.shape)

    # imsave('test.png', transposed)

    # ln = d[b'labels'][1]

    # with open(cifar_dirpath/'batches.meta', 'rb') as fh:
    #     d = pickle.load(fh, encoding='bytes')

    # print(d[b'label_names'][ln])

    # first.reshape(3, 32, 32)


if __name__ == "__main__":
    main()
