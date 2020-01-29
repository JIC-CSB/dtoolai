import click
import dtoolcore

from PIL import Image

from torchvision.transforms.functional import to_tensor

from dtoolai.utils import coerce_to_target_dim
from dtoolai.trained import TrainedTorchModel


def image_fpath_to_model_input(image_fpath, input_format):
    im = Image.open(image_fpath)
    # print(f"Original shape: {im.size}, mode: {im.mode}")
    resized_converted = coerce_to_target_dim(im, input_format)
    as_tensor = to_tensor(resized_converted)
    # Add leading extra dimension for batch size
    return as_tensor[None]


# def diagnose_input(model_input):
#     model_input_np = model_input.squeeze().cpu().numpy()
#     model_input_t = np.transpose(model_input_np, (1, 2, 0))
#     imsave('model_input_t.png', model_input_t)


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

    print(f"Classified {image_fpath} as {result}")





if __name__ == "__main__":
    main()

