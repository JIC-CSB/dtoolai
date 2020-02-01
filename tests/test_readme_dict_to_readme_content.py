def test_nested_dict_as_string():
    expected_repr = """source_ds_name: mnist.test
source_ds_uri: mnist.test
source_ds_uuid: 57deba75-189a-443c-b96d-8005decf05ca
parameters:
  batch_size: 128
  learning_rate: 0.01
  n_epochs: 10
  input_dim: 28
  input_channels: 1
  init_params:
    input_channels: 1
    input_dim: 28
  optimiser_name: SGD
  loss_func: NLLLoss
model_name: simpleScalingCNN
"""

    init_params = {
        "input_channels": 1,
        "input_dim": 28
    }


    parameters_dict = {
        "batch_size": 128,
        "learning_rate": 0.01,
        "n_epochs": 10,
        "input_dim": 28,
        "input_channels": 1,
        "init_params": init_params,
        "optimiser_name": "SGD",
        "loss_func": "NLLLoss"
    }

    readme_dict = {
        "source_ds_name": "mnist.test",
        "source_ds_uri": "mnist.test",
        "source_ds_uuid": "57deba75-189a-443c-b96d-8005decf05ca",
        "parameters": parameters_dict,
        "model_name": "simpleScalingCNN"
    }

    from dtoolai.utils import nested_dict_as_string

    assert nested_dict_as_string(readme_dict) == expected_repr
