import click
import torch
import dtoolcore

import numpy as np

from PIL import Image

import torch.nn as nn
import torchvision.models
from torchvision.transforms.functional import to_tensor

from dtoolai.models import GenNet, ResNet18Retrain
from dtoolai.data import coerce_to_fixed_size_rgb

from imageio import imsave

from dtoolai.trained import TrainedTorchModel


def coerce_to_target_dim(im, input_format):
    """Convert a PIL image to a fixed size and number of channels."""

    mode_map = {
        1: 'L',
        3: 'RGB'
    }

    if im.mode not in mode_map.values():
        raise Exception(f"Unknown image mode: {im.mode}")

    ch, tdimw, tdimh = input_format
    if ch not in mode_map:
        raise Exception(f"Unsupported input format: {input_format}")

    cdimw, cdimh = im.size
    if (cdimw, cdimh) != (tdimw, tdimh):
        im = im.resize((tdimw, tdimh))

    if mode_map[ch] != im.mode:
        im = im.convert(mode_map[ch])

    return im


def image_fpath_to_model_input(image_fpath, input_format):
    im = Image.open(image_fpath)
    # print(f"Original shape: {im.size}, mode: {im.mode}")
    # resized_converted = coerce_to_fixed_size_rgb(im, (256, 256))
    resized_converted = coerce_to_target_dim(im, input_format)
    as_tensor = to_tensor(resized_converted)
    # Add leading extra dimension for batch size
    return as_tensor[None]

def diagnose_input(model_input):
    model_input_np = model_input.squeeze().cpu().numpy()
    model_input_t = np.transpose(model_input_np, (1, 2, 0))
    imsave('model_input_t.png', model_input_t)


@click.command()
@click.argument('model_uri')
@click.argument('image_fpath')
def main(model_uri, image_fpath):

    net = TrainedTorchModel(model_uri)

    dim = net.model_params['input_dim']
    channels = net.model_params['input_channels']
    input_format = [channels, dim, dim]
    model_input = image_fpath_to_model_input(image_fpath, input_format)
    result = net.predict(model_input)
    print(result)





if __name__ == "__main__":
    main()

