def test_resnet_pretrained_retains_properties():

    from dtoolai.models import ResNet18Pretrained

    model = ResNet18Pretrained(5)
    assert model.model_name == "resnet18pretrained"
    assert model.categorical_output == True
