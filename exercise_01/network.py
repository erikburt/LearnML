import random

import numpy as np

"""
    Code mostly taken from: http://neuralnetworksanddeeplearning.com
"""

class Network(object):
    """
        Sizes contains an array of numbers referring the number of neurons in each layer
        ie. A network of 3 layers with an input layer of 10 neurons, a hidden layer of 20 neurons and
        an output layer of 5 neurons would be [10, 20, 5]
    """
    def __init__(self, sizes):
        """Number of layers of this network is the length of sizes passed in"""
        self.num_layers = len(sizes)
        
        """Size of each layer is saved for future reference"""
        self.sizes = sizes
        
        """
            Randomly generate (Gaussian  distributed) biases from 0 to 1 for each neuron in each layer (except the first)
            y -> represents the number of neurons in the current layer
            creates triple nested array like so
            [ 
                [ 
                    [0.1], ..., [0.9] 
                ], 
                [ 
                    [1], ..., [1]
                ]
            ]
        """
        self.biases = [ np.random.randn(y,1) for y in sizes[1:] ]

        """
            Randomly generates (Gaussian  distributed) weights from 0 to 1
            x -> represents the amount of edges of input per neuron in the current layer
            y -> representes the number of neurons in the current layer

            example sizes = [2, 3, 4]
            zip creates an array of tuples out of arrays. zip([2,3], [3,4]) -> [ (2,3), (3,4) ]
            randn(3,2) -> will generate 3 arrays of length 2 representing the edge weights from 
                          the two input neurons to the three neurons in the second layer
            randn(4,3) -> will generate 4 arrays of length 3 representing the edge weights from
                          the three second-layer neurons to the four output neurons
        """
        self.weights = [ np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:]) ]

    """
        Gets the weight for the edge connecting the kth neuron in the ith layer to the 
        jth neuron in the i+1th layer
    """
    def get_weight(self, i, k, j):
        return self.weights[i][k][j]

    """
        The calculation of the network, returns its output given input a
    """
    def feed_forward(self, a):
        for b,w in zip(self.biases, self.weights):
            """
                np.dot(w, a) performs a dot product on the input to the current layer and the
                and the weights associated. This will output a vector the size of the current layer
                and then will add the vector thresholds to that result. Which will give us the input
                for the next layer
            """
            a = self.sigmoid(np.dot(w, a)+b)
        return a


    """
        Trains the neural net by performing stochastic gradient descent on the network
        - Training data is a list of tuples (input, classification) where the size of the input should
        match the size of the first layer, and the classification is the desired output of the NN
        when fed that input.
        - Epochs is how many iterations on the training data you will do
        - Mini_batch_size is the amount of random elements to use in a mini batch of training
        - eta is the learning rate
    """
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None, test_every=1):
        if test_data:
            n_test = len(test_data)
        
        n = len(training_data)

        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) ]

            for mb in mini_batches:
                self.update_mini_batch(mb, eta)

            if test_data and (e+1)%test_every == 0:
                print('Epoch', e, ':', self.evaluate(test_data),'/',n_test)
            else:
                print('Epoch', e, 'complete')
    
    """
        Does the back propagation, the learning portion of the network, does it for a mini batch that is
        decided upon in the stochastic_gradient_descent.

        - Mini batch is in the same format as the training_data input from stochastic_gradient_descent
        - eta is the same as the input from stochastic_gradient_descent

        Note: nabla is the symbol of an upside down delta (triangle). It means the standard derivative.
    """
    def update_mini_batch(self, mini_batch, eta):
        nabla_bias = [ np.zeros(b.shape) for b in self.biases ]
        nabla_weight = [ np.zeros(w.shape) for w in self.weights ]
        
        """ Will create vectors to adjust the weights and biases of our network across the outputs of a mini batch """
        for x, y in mini_batch:
            """ Does the back propagation and calculates the gradient cost of a function returning the deltas of weights and biases"""
            delta_nabla_bias, delta_nabla_weight = self.backprop(x, y)

            """ Adds the delta bias vias to each bias vector """
            nabla_bias = [ (bias + delta_bias) for bias, delta_bias in zip(nabla_bias, delta_nabla_bias) ]

            """ Adds the delta weight vector the each weight vector """
            nabla_weight = [ (weight + delta_weight) for weight, delta_weight in zip(nabla_weight, delta_nabla_weight) ] 
        
        self.biases  = [ b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases,  nabla_bias) ]
        self.weights = [ w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_weight) ]

    """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x.  
    ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``."""
    def backprop(self, training_input, desired_output):
        nabla_b = [ np.zeros(b.shape) for b in self.biases  ]
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]

        """ F E E D F O R W A R D
            this calculates activations and z vectors layer by layer (left to right)
        """
        activation = training_input
        activations = [training_input] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation)+bias # calculates z vector for this layer
            zs.append(z) # append the result so we can calculate a delta later
            activation = self.sigmoid(z) # calculate the actiovation for this layer
            activations.append(activation) # save the activation so we can calculate a delta later

        """ B A C K W A R D P A S S
            this calculates a delta for the weights and biases layer by layer (right to left)
        """
        z = zs[-1]
        sp = self.sigmoid_prime(z)
        delta = self.cost_derivative(activations[-1], desired_output) * sp
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # repeat the above for the rest of the layers
        # the difference is calculating the delta is a little different
        for l in range(2, self.num_layers):
            z = zs[-l] # Get the z vector for right most layer that hasnt been analyzed
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w

    """ Returns the number of correct (input, output) tuples computed by the network """
    def evaluate(self, test_data):
        test_results = [ (np.argmax(self.feed_forward(x)), y) for (x, y) in test_data ]
        return sum(int(x == y) for (x, y) in test_results)

    """ Return the vector of differences for the activations and the desired output """
    def cost_derivative(self, output_activations, desired_output):
        return (output_activations-desired_output)

    """
        This sigmoid activation function of a neuron where z is the input to a neuron
    """
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    """
        This the derivative of the sigmoid function
    """
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
