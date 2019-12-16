import torch
import dtoolcore

import dtoolai.models


MODEL_NAME_LOOKUP = {
    v.model_name: v
    for k, v in dtoolai.models.__dict__.items()
    if isinstance(v, type)
}


class WrappedDataSet(object):

    def __init__(self, uri):
        self.dataset = dtoolcore.DataSet.from_uri(uri)

    def put_overlay(self, overlay_name, overlay):
        self.dataset.put_overlay(overlay_name, overlay)

    @property
    def name(self):
        return self.dataset.name

    @property
    def uri(self):
        return self.dataset.uri

    @property
    def uuid(self):
        return self.dataset.uuid


class TrainedTorchModel(WrappedDataSet):

    def __init__(self, uri, model_cls=None):
        super().__init__(uri)

        model_name = self.dataset.get_annotation('model_name')
        if model_name.startswith('dtoolai'):
            _, name_lookup_key = model_name.split('.')
            # print(f"Lookup {name_lookup_key}")
            model_cls = MODEL_NAME_LOOKUP[name_lookup_key]

        model_params = self.dataset.get_annotation("model_parameters")
        self.model = model_cls(**model_params['init_params'])
        self.model_params = model_params

        cat_encoding = self.dataset.get_annotation("category_encoding")
        self.cat_decoding = {n: cat for cat, n in cat_encoding.items()}

        idn = dtoolcore.utils.generate_identifier('model.pt')
        state_abspath = self.dataset.item_content_abspath(idn)
        self.model.load_state_dict(torch.load(state_abspath, map_location='cpu'))
        self.model.eval()

    @classmethod
    def from_uri(cls, uri):
        torchmodel = cls(uri)
        return torchmodel

    def predict(self, model_input):

        with torch.no_grad():
            output = self.model(model_input)

        cat_n = output.squeeze().numpy().argmax()

        return self.cat_decoding[cat_n]
