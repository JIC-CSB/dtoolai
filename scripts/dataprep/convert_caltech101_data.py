import shutil

from pathlib import Path

import click


README_TEMPLATE="""---
dataset_name: Caltech 101 images subset
project: dtoolAI demonstration datasets
authors:
  - Fei-Fei Li
  - Marco Andreetto
  - Marc 'Aurelio Ranzato
reference: |
  L. Fei-Fei, R. Fergus and P. Perona. One-Shot learning of object
  categories. IEEE Trans. Pattern Recognition and Machine Intelligence.
origin: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
"""

@click.command()
@click.argument('base_dirpath')
@click.argument('staging_dirpath')
@click.option('--categories', default='llama,hedgehog')
def main(base_dirpath, staging_dirpath, categories):

    base_dirpath = Path(base_dirpath)
    staging_dirpath = Path(staging_dirpath)
    categories_list = categories.split(',')

    for category in categories_list:
        shutil.copytree(base_dirpath/category, staging_dirpath/category)


if __name__ == "__main__":
    main()
