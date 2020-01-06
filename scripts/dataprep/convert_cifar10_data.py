import pickle

from pathlib import Path

import click
import dtoolcore

import numpy as np

from dtool_utils.quick_dataset import QuickDataSet


README_TEMPLATE = """---
dataset_name: CIFAR-10 (10 category subset of 80 million tiny images)
project: dtoolAI demonstration datasets
authors:
  - Alex Krizhevsky
  - Vinod Nair
  - Geoffrey Hinton
origin: https://www.cs.toronto.edu/~kriz/cifar.html
reference: |
  Learning Multiple Layers of Features from Tiny Images 
  <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>
usetype: {usetype}
"""


def data_and_labels_from_fname_list(cifar_dirpath, fname_list):

    def load_batch(batch_name):
        with open(cifar_dirpath/batch_name, 'rb') as fh:
            cifar_dict = pickle.load(fh, encoding='bytes')
        return cifar_dict[b'data'], cifar_dict[b'labels']

    loaded_batches = [load_batch(bn) for bn in fname_list]

    data_arrays, label_arrays = zip(*loaded_batches)

    data = np.concatenate(data_arrays)
    labels = np.concatenate(label_arrays)

    return data, labels


def convert_cifar10_data(cifar_dirpath, output_base_uri, output_name, mode='train'):

    if mode == 'train':
        batches = [f'data_batch_{n}' for n in range(1, 6)]
    elif mode == 'test':
        batches = ['test_batch']

    data, labels = data_and_labels_from_fname_list(cifar_dirpath, batches)

    with QuickDataSet(output_base_uri, output_name) as qds:
        data_abspath = qds.staging_fpath('cifar.npy')
        np.save(data_abspath, data)
        labels_fpath = qds.staging_fpath('labels.npy')
        np.save(labels_fpath, labels)

    tensor_idn = dtoolcore.utils.generate_identifier('cifar.npy')
    qds.put_annotation("tensor_file_idn", tensor_idn)
    
    image_dim = [3, 32, 32]
    qds.put_annotation("image_dimensions", image_dim)
    qds.put_annotation("dtoolAI.inputtype", "TensorDataSet")
    qds.put_readme(README_TEMPLATE.format(usetype=mode))



@click.command()
@click.argument('cifar_base_dirpath')
@click.argument('output_base_uri')
@click.argument('output_name')
@click.option('--mode', default='train')
def main(cifar_base_dirpath, output_base_uri, output_name, mode):
    
    cifar_dirpath = Path(cifar_base_dirpath)
    convert_cifar10_data(cifar_dirpath, output_base_uri, output_name, mode)


if __name__ == "__main__":
    main()
