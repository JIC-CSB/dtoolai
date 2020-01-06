import os

import click

from pathlib import Path

# from dtool_utils import DerivedDataSet
from dtool_utils.quick_dataset import QuickDataSet


@click.command()
@click.argument('dirpath')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(dirpath, output_base_uri, output_name):

    categories = [d for d in os.listdir(dirpath)]

    def relpath_from_srcpath(srcpath, cat):
        return cat + '/' + os.path.basename(srcpath)

    items_to_include = {}
    for cat in categories:
        srcpath_iter = (Path(dirpath) / cat).iterdir()
        items_to_include.update({
            srcpath: (relpath_from_srcpath(srcpath, cat), cat)
            for srcpath in srcpath_iter
        })

    category_encoding = {c: n for n, c in enumerate(categories)}
    with QuickDataSet(output_base_uri, output_name) as output_ds:
        for srcpath, (relpath, cat) in items_to_include.items():
            handle = output_ds.put_item(srcpath, relpath)
            output_ds.add_item_metadata(handle, 'category', cat)
        output_ds.put_annotation('category_encoding', category_encoding)
        output_ds.put_annotation("dtoolAI.inputtype", "ImageDataSet")


if __name__ == "__main__":
    main()
