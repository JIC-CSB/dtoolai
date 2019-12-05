
import json

class Parameters(object):

    def __init__(self, **kwargs):
        self.parameter_dict = dict(kwargs)

    @classmethod
    def from_dict(cls, input_dict):

        parameters = cls()
        parameters.parameter_dict = input_dict

        return parameters

    @classmethod
    def from_json_string(cls, json_string):

        json_dict = json.loads(json_string)

        parameters = cls()
        parameters.parameter_dict.update(json_dict)

        return parameters

    def __getattr__(self, name):
        return self.parameter_dict[name]

    def __setitem__(self, key, value):
        self.parameter_dict[key] = value

    def __getitem__(self, key):
        return self.parameter_dict[key]

    def __repr__(self):
        return str(self.parameter_dict)
