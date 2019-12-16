import click
import dtoolcore


@click.command()
@click.argument('source_ds_uri')
@click.argument('dest_ds_uri')
@click.argument('annotation_name')
def main(source_ds_uri, dest_ds_uri, annotation_name):


    source_ds = dtoolcore.DataSet.from_uri(source_ds_uri)
    dest_ds = dtoolcore.DataSet.from_uri(dest_ds_uri)

    annotation = source_ds.get_annotation(annotation_name)
    dest_ds.put_annotation(annotation_name, annotation)



if __name__ == "__main__":
    main()
