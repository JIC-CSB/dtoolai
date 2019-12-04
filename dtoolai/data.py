import torch.utils.data
import dtoolcore

import numpy as np

from sklearn.model_selection import train_test_split


class WrappedDataSet(torch.utils.data.Dataset):

    def __init__(self, uri):
        self.dataset = dtoolcore.DataSet.from_uri(uri)

    @property
    def name(self):
        return self.dataset.name

    @property
    def uri(self):
        return self.dataset.uri

    @property
    def uuid(self):
        return self.dataset.uuid


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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=0)

        if test:
            self.tensor = self.X_test
            self.labels = self.y_test
        else:
            self.tensor = self.X_train
            self.labels = self.y_train

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
