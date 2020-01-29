def test_mnist_examples():

    mnist_train_uri = "http://bit.ly/2uqXxrk"
    mnist_test_uri = "http://bit.ly/2NVFGQd"

    from dtoolai.data import TensorDataSet

    tds = TensorDataSet(mnist_train_uri)
    assert tds.name == "mnist.train"
    assert len(tds) == 60000

    tds = TensorDataSet(mnist_test_uri)
    assert tds.name == "mnist.test"
    assert len(tds) == 10000
