import os

import numpy as np

from . import tmp_dir_fixture
from . import TEST_SAMPLE_DATA



def test_tensor_dataset_functional():

    from dtoolai.data import TensorDataSet

    tds_uri = os.path.join(TEST_SAMPLE_DATA, "example_tensor_dataset")

    tds = TensorDataSet(tds_uri)
    assert tds.name == "example_tensor_dataset"
    assert tds.uuid == "6b6f9a0e-8547-4903-9090-6dcfc6abdf83"
    assert len(tds) == 100

    data, label = tds[0]
    assert data.shape == (1, 9, 9)
    assert data[0][0][0] == 0
    assert label == 0

    assert tds.input_channels == 1
    assert tds.dim == 9


def test_image_dataset_functional():

    from dtoolai.data import ImageDataSet

    ids_uri = os.path.join(TEST_SAMPLE_DATA, "example_image_dataset")

    ids = ImageDataSet(ids_uri)
    assert ids.name == "example_image_dataset"
    assert ids.uuid == "bb27e945-b789-45d3-9b00-4808aa09ea03"
    assert len(ids) == 6

    assert ids.input_channels == 3
    assert ids.dim == 256

    im, label = ids[0]
    assert isinstance(im, np.ndarray)
    assert label == 2
    

def test_create_tensor_dataset_from_arrays(tmp_dir_fixture):
    pass


