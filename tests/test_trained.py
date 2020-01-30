import os

from PIL import Image

from . import TEST_SAMPLE_DATA


def test_trained_model():

    from dtoolai.trained import TrainedTorchModel

    model = TrainedTorchModel("http://bit.ly/2tbPzSB")

    assert model.model_name == "dtoolai.simpleScalingCNN"
    assert model.model_params["input_dim"] == 28
    assert model.model_params["input_channels"] == 1

    assert "mnist" in model.get_readme_content()

    image_fpath = os.path.join(TEST_SAMPLE_DATA, "non_mnist_8.png")
    im = Image.open(image_fpath)

    prediction = model.convert_and_predict(im)
    assert prediction == 8
    
