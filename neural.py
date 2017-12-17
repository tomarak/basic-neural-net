import numpy


DEFAULT_CYCLES = 1000


class NeuralNetwork(object):
    """Define the neural network base class."""
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        """Initialize the layer sizes and the weights"""
        # layer parameters
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        # weights
        self.weight_matrix_a = numpy.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.weight_matrix_b = numpy.random.randn(self.hidden_layer_size, self.output_layer_size)

    def propagate(self, input_data):
        """
        Forward feed data into network.

        First, we get the dot product of the inputdata and the first matrix of weights. Then, we apply the sigmoid
        activation function to the dotproduct. We repeat this process witht he second set of weights.
        """
        # Dotproduct of the input and the first set of weights
        self.z = numpy.dot(input_data, self.weight_matrix_a)
        # Apply the sigmoid
        self.z2 = self.sigmoid(self.z)
        self.z3 = numpy.dot(self.z2, self.weight_matrix_b)
        return self.sigmoid(self.z3)

    def sigmoid(self, s):
        """Return the sigmoid of an input value."""
        # activation function
        return 1/(1+numpy.exp(-s))

    def sigmoidPrime(self, s):
        """Return the derivative sigmoid value."""
        return s * (1 - s)

    def backpropagate(self, input_data, output_data, output):
        # backpropagate propgate through the network
        # error in output
        self.output_error = output_data - output
        # applying derivative of sigmoid to error
        self.output_delta = self.output_error*self.sigmoidPrime(output)

        # z2 error: how much our hidden layer weights contributed to output error
        self.z2_error = self.output_delta.dot(self.weight_matrix_b.T)
        # applying derivative of sigmoid to z2 error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

        # adjusting first set (input --> hidden) weights
        self.weight_matrix_a += input_data.T.dot(self.z2_delta)
        # adjusting second set (hidden --> output) weights
        self.weight_matrix_b += self.z2.T.dot(self.output_delta)

    def train_network(self, cycles=DEFAULT_CYCLES):
        """Train the network."""
        for i in range(cycles):
            print("Input (scaled): \n" + str(input_data))
            print("Actual Output: \n" + str(output_data))
            print("Predicted Output: \n" + str(self.propagate(input_data)))

            print("Loss: \n" + str(numpy.mean(numpy.square(output_data - self.propagate(input_data)))))
            print("\n")
            self.train(input_data, output_data)

    def train(self, input_data, output_data):
        output = self.propagate(input_data)
        self.backpropagate(input_data, output_data, output)

    def saveWeights(self):
        """Save the randomly generated weights to a text file."""
        numpy.savetxt("weights_a.txt", self.weight_matrix_a, fmt="%s")
        numpy.savetxt("weights_b.txt", self.weight_matrix_b, fmt="%s")

    def predict(self, xPredicted):
        """Determine the predicted value."""
        xPredicted = xPredicted/numpy.amax(xPredicted, axis=0)
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.propagate(xPredicted)))