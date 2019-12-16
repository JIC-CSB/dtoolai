import logging

import torch.utils.data
import dtoolcore

import numpy as np

from sklearn.model_selection import train_test_split

from PIL import Image

from dtool_utils.quick_dataset import QuickDataSet


class WrappedDataSet(torch.utils.data.Dataset):

    def __init__(self, uri):
        self.dataset = dtoolcore.DataSet.from_uri(uri)

    def put_overlay(self, overlay_name, overlay):
        self.dataset.put_overlay(overlay_name, overlay)

    @property
    def name(self):
        return self.dataset.name

    @property
    def uri(self):
        return self.dataset.uri

    @property
    def uuid(self):
        return self.dataset.uuid


def coerce_to_fixed_size_rgb(im, target_dim):
    """Convert a PIL image to a fixed size and 3 channel RGB format."""

    if im.mode not in ['RGB', 'L']:
        raise Exception(f"Unknown image mode: {im.mode}")

    resized_im = im.resize(target_dim)

    if im.mode is 'RGB':
        return resized_im

    return resized_im.convert('RGB')


class ImageDataSet(WrappedDataSet):
    
    def __init__(self, uri, usetype='train'):
        super().__init__(uri)

        self.loaded_images = {}
        self.cat_lookup = self.dataset.get_overlay('category')
        self.cat_encoding = self.dataset.get_annotation('category_encoding')
        self.image_dim = 256, 256

        try:
            usetype_overlay = self.dataset.get_overlay('usetype')

            self.identifiers = [
                idn for idn in self.dataset.identifiers
                if usetype_overlay[idn] == usetype
            ]
        except dtoolcore.DtoolCoreKeyError:
            self.identifiers = self.dataset.identifiers

        self.idn_lookup = {n: idn for n, idn in enumerate(self.identifiers)}

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, index):
        
        idn = self.idn_lookup[index]

        if idn not in self.loaded_images:
            logging.debug(f"Loading {self.dataset.item_content_abspath(idn)}")
            im = Image.open(self.dataset.item_content_abspath(idn))
            # print(f"Original shape: {im.size}, mode: {im.mode}")
            resized_converted = coerce_to_fixed_size_rgb(im, self.image_dim)
            channels_first = np.moveaxis(np.array(resized_converted), 2, 0)
            self.loaded_images[idn] = channels_first.astype(np.float32) / 255

        return self.loaded_images[idn], self.cat_encoding[self.cat_lookup[idn]]

    @property
    def input_channels(self):
        return 3

    @property
    def dim(self):
        return self.image_dim[0]


# TODO - FIX THE OBJECT TYPE
class TensorDataSet(WrappedDataSet):

    def __init__(self, uri, test=False):
        super().__init__(uri)

        tensor_file_idn = self.dataset.get_annotation("tensor_file_idn")
        npy_fpath = self.dataset.item_content_abspath(tensor_file_idn)
        self.X = np.load(npy_fpath, mmap_mode=None)

        labels_idn = dtoolcore.utils.generate_identifier("labels.npy")
        labels_fpath = self.dataset.item_content_abspath(labels_idn)
        self.y = np.load(labels_fpath, mmap_mode=None)

        self.image_dim = self.dataset.get_annotation("image_dimensions")

        self.tensor = self.X
        self.labels = self.y

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, index):
        raw = self.tensor[index]
        scaledfloat = raw.astype(np.float32) / 255

        label = self.labels[index]

        return scaledfloat.reshape(*self.image_dim), int(label)

    @property
    def input_channels(self):
        return self.image_dim[0]

    @property
    def dim(self):
        return self.image_dim[1]


def create_tensor_dataset_from_arrays(
    output_base_uri, output_name, data_array, label_array, image_dim, readme_content
):
    with QuickDataSet(output_base_uri, output_name) as qds:
        data_fpath = qds.staging_fpath('data.npy')
        np.save(data_fpath, data_array)
        labels_fpath = qds.staging_fpath('labels.npy')
        np.save(labels_fpath, label_array)

    data_idn = dtoolcore.utils.generate_identifier('data.npy')
    qds.put_annotation("tensor_file_idn", data_idn)
    qds.put_annotation("image_dimensions", image_dim)
    qds.put_annotation("dtoolAI.inputtype", "TensorDataSet")

    qds.put_readme(readme_content)
