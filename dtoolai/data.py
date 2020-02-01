import logging

import torch.utils.data
import dtoolcore

import numpy as np

from PIL import Image


class WrappedDataSet(torch.utils.data.Dataset):
    """Subclass of pytorch Dataset that provides dtool DataSet methods.

    Args:

        uri: URI for enclosed dtool DataSet.
    """

    def __init__(self, uri):
        self.dataset = dtoolcore.DataSet.from_uri(uri)

    def put_overlay(self, overlay_name, overlay):
        self.dataset.put_overlay(overlay_name, overlay)

    def get_annotation(self, annotation_name):
        return self.dataset.get_annotation(annotation_name)

    @property
    def name(self):
        return self.dataset.name

    @property
    def uri(self):
        return self.dataset.uri

    @property
    def uuid(self):
        return self.dataset.uuid


def scaled_float_array_to_pil_image(array):
    """Convert an array of floats to a PIL image.
    
    Args:
        array (np.ndarray): Array representing an image. Expected to be float
            and normalised between 0 and 1.
    
    """

    intarray = (255 * array).astype(np.uint8)

    if len(array.shape) > 3:
        raise ValueError(f"Can't handle array of shape {array.shape}")

    if len(array.shape) == 2:
        return Image.fromarray(intarray)
    elif len(array.shape) == 3:
        intarray = np.transpose(intarray, (1, 2, 0))
        channels = intarray.shape[2]
        if channels == 1:
            return Image.fromarray(intarray.squeeze())
        elif channels == 3:
            return Image.fromarray(intarray)
        else:
            raise ValueError(f"Can't handle image with {channels} channels")
    else:
        raise ValueError(f"Can't handle array with shape {array.shape}")



def coerce_to_fixed_size_rgb(im, target_dim):
    """Convert a PIL image to a fixed size and 3 channel RGB format."""

    if im.mode not in ['RGB', 'L']:
        raise Exception(f"Unknown image mode: {im.mode}")

    resized_im = im.resize(target_dim)

    if im.mode is 'RGB':
        return resized_im

    return resized_im.convert('RGB')


class ImageDataSet(WrappedDataSet):
    """Class allowing a collection of images annotated with categories to be
    used as both a Pytorch Dataset and a dtool DataSet."""
    
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


class TensorDataSet(WrappedDataSet):
    """Class that allows numpy arrays to be accessed as both a pytorch
    Dataset and a dtool DataSet."""

    def __init__(self, uri):
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
        """The number of channels each tensor provides."""
        return self.image_dim[0]

    @property
    def dim(self):
        """The linear dimensions of the tensor, e.g. it is dim x dim in shape.
        """
        return self.image_dim[1]


def create_tensor_dataset_from_arrays(
    output_base_uri, output_name, data_array, label_array, image_dim, readme_content
):
    """Create a dtool DataSet with the necessary annotations to be used as a
    TensorDataSet.

    Args:
        output_base_uri: The base URI where the dataset will be created.
        output_name: The name for the output dataset.
        data_array (ndarray): The numpy array holding data.
        label_array (ndarray): The numpy array holding labels.
        image_dim (tuple): Dimensions to which input images should be reshaped.
        readme_content (string): Content that will be used to create README.yml
            in the created dataset.

    Returns:
        URI: The URI of the created dataset
    
    """

    with dtoolcore.DataSetCreator(output_name, output_base_uri) as qds:
        data_fpath = qds.prepare_staging_abspath_promise('data.npy')
        np.save(data_fpath, data_array)
        labels_fpath = qds.prepare_staging_abspath_promise('labels.npy')
        np.save(labels_fpath, label_array)

    data_idn = dtoolcore.utils.generate_identifier('data.npy')
    qds.put_annotation("tensor_file_idn", data_idn)
    qds.put_annotation("image_dimensions", image_dim)
    qds.put_annotation("dtoolAI.inputtype", "TensorDataSet")

    qds.put_readme(readme_content)

    return qds.uri
