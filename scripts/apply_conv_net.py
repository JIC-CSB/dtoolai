
import click
import dtoolcore
import torch

import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor


from cnn import Net
from data import TensorDataSet

from ruamel.yaml import YAML


class TorchModel(object):

    @classmethod
    def from_uri(cls, uri, base_model_func):
        self = cls()
        self.dataset = dtoolcore.DataSet.from_uri(uri)

        # yaml = YAML()
        # parsed_readme = yaml.load(self.dataset.get_readme_content())
        # self.params = parsed_readme['parameters']

        # base_model = base_model_func(self.params['input_channels'], self.params['output_channels'])

        base_model = base_model_func()

        model_idn = dtoolcore.utils.generate_identifier("model.pt")
        model_fpath = self.dataset.item_content_abspath(model_idn)
        base_model.load_state_dict(torch.load(model_fpath, map_location='cpu'))
        self.base_model = base_model

        return self

    @property
    def model(self):
        return self.base_model

    @property
    def name(self):
        return self.dataset.name

    def predict(self, model_input):

        with torch.no_grad():
            output = self.base_model(model_input)

        return output.squeeze().numpy()


def test(model, dl, tds):

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in dl:
                Y_pred = model(data)
                test_loss += F.nll_loss(Y_pred, label).item()
                pred = Y_pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        print("Test loss:", test_loss)
        print(f"{correct}/{len(tds)} correct")


@click.command()
@click.argument('model_uri')
@click.argument('test_data_uri')
def main(model_uri, test_data_uri):

    cnnmodel = TorchModel.from_uri(model_uri, Net)

    tds = TensorDataSet(test_data_uri)
    dl = DataLoader(tds, batch_size=64, shuffle=True)xw

    data, labels = next(iter(dl))

    result = cnnmodel.predict(data)

    predicted_digits = np.argmax(result, axis=1)

    print(labels.numpy() == predicted_digits)
    # model = Net()
    # model.load_state_dict(torch.load("model.pt"))

    # test_tds = TensorDataSet(test_data_uri, test=True)
    # test_dl = DataLoader(test_tds, batch_size=64, shuffle=True)

    # test(model, test_dl, test_tds)


if __name__ == "__main__":
    main()
