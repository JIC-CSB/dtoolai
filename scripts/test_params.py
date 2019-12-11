import click

from dtoolai.parameters import Parameters


@click.command()
@click.option('--params')
def main(params):

    test_params = Parameters(
        greeting="hello",
        n_bananas=12,
        fraction_useful=0.2
    )
    if params:
        test_params.update_from_comma_separated_string(params)

    update_string = "n_bananas=15"
    test_params.update_from_comma_separated_string(update_string)

    print(test_params)



if __name__ == "__main__":
    main()
