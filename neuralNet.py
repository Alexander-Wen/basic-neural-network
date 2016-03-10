import numpy as np

# input training data
# hours sleep, hours studying
x = np.array(([3,5], [5,1], [10,2]), dtype=float)
# result training data
# grade
y = np.array(([75],[82],[93]), dtype=float)
# scale inputs so they are in the same format
x = x/np.amax(x, axis=0)
y = y/100

class Neural_Network(object):
    def __init__(self):
        # Define Hyperparamters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights of the synapses
        self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, x):
        # Propogate inputs through network
        # matrix multiply input by the synapse weights
        self.z2 = np.dot(x, self.w1)
        # apply sigmoid activation on the result
        self.a2 = self.sigmoid(self.z2)
        # matrix multiply with second layer of synapses
        self.z3 = np.dot(self.a2, self.w2)
        # apply sigmoid activation
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # apply sigmoid activation function
        return 1/(1+np.exp(-z))

NN = Neural_Network()
yHat = NN.forward(x)
print yHat
print y
