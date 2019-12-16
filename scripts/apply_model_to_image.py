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


def image_fpath_to_model_input(image_fpath):
    im = Image.open(image_fpath)
    # print(f"Original shape: {im.size}, mode: {im.mode}")
    resized_converted = coerce_to_fixed_size_rgb(im, (256, 256))
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
    model_input = image_fpath_to_model_input(image_fpath)
    result = net.predict(model_input)
    print(result)
    # model = ResNet18Retrain(2)


    # ds = dtoolcore.DataSet.from_uri(model_uri)
    # cat_encoding = ds.get_annotation("category_encoding")
    # cat_decoding = {n: cat for cat, n in cat_encoding.items()}
    # # model_params = ds.
    # idn = dtoolcore.utils.generate_identifier('model.pt')
    # state_abspath = ds.item_content_abspath(idn)
    # model.load_state_dict(torch.load(state_abspath, map_location='cpu'))
    # model.eval()

    # model_input = image_fpath_to_model_input(image_fpath)
    # with torch.no_grad():
    #     output = model(model_input)

    # cat_n = output.squeeze().numpy().argmax()
    # print(f"Classified as: {cat_decoding[cat_n]}")




if __name__ == "__main__":
    main()

