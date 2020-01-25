import pytest

def test_update_from_command_separated_string():

    from dtoolai.parameters import Parameters

    test_params = Parameters(
        greeting="hello",
        n_bananas=12,
        fraction_useful=0.2
    )

    assert test_params.n_bananas == 12

    update_string = "n_bananas=15,fraction_useful=0.3"
    test_params.update_from_comma_separated_string(update_string)

    assert test_params.n_bananas == 15
    assert test_params.fraction_useful == 0.3


def test_strict_behaviour():

    from dtoolai.parameters import Parameters

    test_params = Parameters(
        greeting="hello",
        n_bananas=12,
        fraction_useful=0.2
    )

    with pytest.raises(NameError):
        test_params.update_from_comma_separated_string("foo=bar")

    test_params.update_from_comma_separated_string("foo=bar", strict=False)
    assert test_params.foo == "bar"

