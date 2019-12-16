import click

import numpy as np

from imageio import imsave

from dtoolai.data import TensorDataSet


@click.command()
@click.argument('tensor_dataset_uri')
@click.argument('output_prefix')
def main(tensor_dataset_uri, output_prefix):

    tds = TensorDataSet(tensor_dataset_uri)

    im, label = tds[30]

    im_tr = np.transpose(im, (1, 2, 0))
    output_fpath = f"{output_prefix}_{label}.png"
    imsave(output_fpath, im_tr)


if __name__ == "__main__":
    main()
