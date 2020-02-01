import click

import dtoolcore

import dtoolai
from dtoolai.data import TensorDataSet, ImageDataSet


DSTYPE_LOOKUP = {
    v.__name__: v
    for k, v in dtoolai.data.__dict__.items()
    if isinstance(v, type)
}


class MetaDataSet(object):

    @classmethod
    def from_uri(cls, uri):
        ds = dtoolcore.DataSet.from_uri(uri)
        dstype = ds.get_annotation('dtoolAI.inputtype')
        ds_cls = DSTYPE_LOOKUP[dstype]

        return ds_cls(uri)


@click.command()
@click.argument('input_dataset_uri')
def main(input_dataset_uri):

    ds = MetaDataSet.from_uri(input_dataset_uri)
    print(len(ds))


if __name__ == "__main__":
    main()
