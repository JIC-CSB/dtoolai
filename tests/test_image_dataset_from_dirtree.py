import os

from . import tmp_dir_fixture
from . import TEST_SAMPLE_DATA


def test_image_dataset_from_dirtree(tmp_dir_fixture):

    dirtree_dirpath = os.path.join(TEST_SAMPLE_DATA, "image_dirtree")

    from dtoolai.utils import image_dataset_from_dirtree

    uri = image_dataset_from_dirtree(dirtree_dirpath, tmp_dir_fixture, "imageds")

    assert uri.startswith("file://")
    assert uri.endswith(os.path.join(tmp_dir_fixture, "imageds"))

    from dtoolai.data import ImageDataSet

    ids = ImageDataSet(uri)

    assert len(ids) == 6
    assert set(ids.cat_encoding.keys()) == set(['car', 'mug', 'chair'])


