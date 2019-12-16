import click

from dtoolai.trained import TrainedTorchModel


@click.command()
@click.argument('model_uri')
def main(model_uri):

    net = TrainedTorchModel(model_uri)



if __name__ == "__main__":
    main()
