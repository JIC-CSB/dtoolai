import random
from collections import defaultdict

import click

from dtoolai.data import ImageDataSet


def annotate_by_random_selection(ds, frac_test=0.2):

    idns_by_cat = defaultdict(list)
    for idn, cat in ds.cat_lookup.items():
        idns_by_cat[cat].append(idn)

    usetype_overlay = {}
    for idns in idns_by_cat.values():
        n_test = int(frac_test * len(idns))
        random.shuffle(idns)
        for idn in idns[:n_test]:
            usetype_overlay[idn] = 'test'
        for idn in idns[n_test:]:
            usetype_overlay[idn] = 'train'

    ds.put_overlay('usetype', usetype_overlay)


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    ids = ImageDataSet(dataset_uri)

    annotate_by_random_selection(ids)


if __name__ == "__main__":
    main()
