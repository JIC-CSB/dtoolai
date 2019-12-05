import shutil

from pathlib import Path

import click


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
