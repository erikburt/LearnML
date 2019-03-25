from network import Network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

size = [784,30,10]
net = Network(size)

net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data=test_data, test_every=5)


