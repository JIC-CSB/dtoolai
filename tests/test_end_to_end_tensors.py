import os

import dtoolcore
import pytest
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from . import tmp_dir_fixture
from . import TEST_SAMPLE_DATA


@pytest.mark.slow
def test_end_to_end_tensors_with_mnist(tmp_dir_fixture):

    data_cache_dirpath = os.path.join(TEST_SAMPLE_DATA, "torch")
    mnist_torch = torchvision.datasets.MNIST(data_cache_dirpath, download=True)
    data = mnist_torch.data.numpy().reshape(60000, -1)
    labels = mnist_torch.targets
    # mnist_numpy_uri = os.path.join(TEST_SAMPLE_DATA, "mnist.numpy")
    # mnist_ds = dtoolcore.DataSet.from_uri(mnist_numpy_uri)
    # data_dtoolcore.utils.generate_identifier("data.npy")
    # data_fpath = mnist_ds.item_content_abspath

    # mnist_train_numpy_dirname = "mnist_train_numpy"
    # data_fpath = os.path.join(TEST_SAMPLE_DATA, mnist_train_numpy_dirname, "data.npy")
    # labels_fpath = os.path.join(TEST_SAMPLE_DATA, mnist_train_numpy_dirname, "labels.npy")

    # data = np.load(data_fpath)
    assert data.shape == (60000, 784)
    # labels = np.load(labels_fpath)
    assert labels.shape == (60000,)

    from dtoolai.data import create_tensor_dataset_from_arrays

    create_tensor_dataset_from_arrays(
        tmp_dir_fixture,
        "mnist.train",
        data,
        labels,
        (1, 28, 28),
        ""
    )

    from dtoolai.data import TensorDataSet

    mnist_train_uri = os.path.join(tmp_dir_fixture, "mnist.train")
    tds = TensorDataSet(mnist_train_uri)
    assert len(tds) == 60000

    dl = DataLoader(tds, batch_size=128, shuffle=True)
    data, labels = next(iter(dl))
    assert data.shape == (128, 1, 28, 28)

    from dtoolai.models import GenNet
    init_params = dict(input_channels=tds.input_channels, input_dim=tds.dim)
    model = GenNet(**init_params)
    loss_fn = F.nll_loss
    optimiser = optim.SGD(model.parameters(), lr=0.01)

    from dtoolai.utils import train
    train(model, dl, optimiser, loss_fn, 3)

    # FIXME - should be hosted or something
    model.eval()
    tds_test = TensorDataSet("http://bit.ly/2NVFGQd")
    dl_test = DataLoader(tds_test, batch_size=128)
    assert len(tds_test) == 10000
    correct = 0
    with torch.no_grad():
        for data, label in dl_test:
            Y_pred = model(data)
            pred = Y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    assert correct >= 6000


