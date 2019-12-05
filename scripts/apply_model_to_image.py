import click
import torch
import dtoolcore

from PIL import Image

from torchvision.transforms.functional import to_tensor

from dtoolai.models import GenNet


@click.command()
@click.argument('image_fpath')
def main(image_fpath):


    model = GenNet(1, 28)
    model_uri = 'scratch/models/conv2dmnist'

    ds = dtoolcore.DataSet.from_uri(model_uri)
    idn = dtoolcore.utils.generate_identifier('model.pt')
    state_abspath = ds.item_content_abspath(idn)

    model.load_state_dict(torch.load(state_abspath, map_location='cpu'))

    im = Image.open(image_fpath)
    im_resized = im.resize((28, 28))
    im_converted = im_resized.convert('L')

    im_converted.save('i.png')

    single_input = to_tensor(im_converted)
    model_input = single_input[None]

    with torch.no_grad():
        output = model(model_input)

    print(output.squeeze().numpy())
    print(output.squeeze().numpy().argmax())


if __name__ == "__main__":
    main()

