import json

class Parameters(object):
    """Class holding key/value parameter data."""

    def __init__(self, **kwargs):
        self.parameter_dict = dict(kwargs)
        self.param_types = {k: type(v) for k, v in kwargs.items()}

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

    @classmethod
    def from_comma_separated_string(cls, raw_string):

        if raw_string is None:
            return cls()

        param_dict = dict(p.split('=') for p in raw_string.split(','))

        parameters = cls()
        parameters.parameter_dict.update(param_dict)

        return parameters

    def update_from_comma_separated_string(self, raw_string, strict=True):
        param_dict = dict(p.split('=') for p in raw_string.split(','))

        for k, v in param_dict.items():
            
            if k not in self.parameter_dict and strict:
                raise NameError(f"Parameter name {k} not known and strict=True")

            param_type = self.param_types.get(k, str)
            self.parameter_dict[k] = param_type(v)

    def set_defaults(self, **kwargs):
        defaults = dict(kwargs)
        defaults.update(self.parameter_dict)
        self.parameter_dict = defaults

    def __getattr__(self, name):
        return self.parameter_dict[name]

    def __setitem__(self, key, value):
        self.parameter_dict[key] = value

    def __getitem__(self, key):
        return self.parameter_dict[key]

    def __repr__(self):
        return str(self.parameter_dict)
