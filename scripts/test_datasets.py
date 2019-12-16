import click

import numpy as np

from dtoolai.data import TensorDataSet, ImageDataSet

from torch.utils.data import DataLoader, random_split

from imageio import imsave

from sklearn.model_selection import train_test_split


def test_tensor_dataset(dataset_uri):
    tds = TensorDataSet(dataset_uri)

    train_test_split(tds, [60000, 10000])
    dl = DataLoader(tds, batch_size=64)

    tensor, label = next(iter(dl))

    print(tensor.shape)
    print(label.shape)



def test_image_dataset(dataset_uri):
    ids = ImageDataSet(dataset_uri, usetype='test')
    # dl = DataLoader(ids, batch_size=4)

    print(f"Dataset has {len(ids)} items")

    image, label = ids[7]

    # image, label = ids[8]
    print(image.shape)
    # for image, label in iter(ids):
    #     print(image.shape)

    # for tensor, label in iter(dl):
    #     print(tensor.shape, tensor.dtype)
    #     print(label.shape)

    tr = np.transpose(image, axes=(1, 2, 0))
    imsave('tr.png', tr)


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    test_tensor_dataset(dataset_uri)
    # test_image_dataset(dataset_uri)



if __name__ == "__main__":
    main()
