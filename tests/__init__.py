"""Test fixtures."""

import os
import shutil
import tempfile
from contextlib import contextmanager

import pytest


_HERE = os.path.dirname(__file__)
TEST_SAMPLE_DATA = os.path.join(_HERE, "data")


@pytest.fixture
def tmp_dir_fixture(request):
    d = tempfile.mkdtemp()

    @request.addfinalizer
    def teardown():
        shutil.rmtree(d)
    return d
