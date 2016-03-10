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

    def costFunction(self, x, y):
        self.yHat = self.forward(x)
        # cost is defined as sum of (0.5*(y-yHat)^2)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, x, y):
        # compute derivative with respect to w1 and w2
        # forward propogate to get inital guess
        self.yHat = self.forward(x)

        # cost function is defined as
        # sum(0.5(y-yHat)^2)
        # we need to minimize cost by using gradient descent
        # we take the derivative of the cost
        # d(0.5(y-yHat)^2)/dw2
        # (y-yHat)(-dyHat/dw2)
        # -(y-yHat)(dyHat/dz3)(dz3/dw2)
        # -(y-yHat)f'(z3)(dz3/dw2)
        # -(y-yHat)f'(z3) is delta3
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        # dz3/dw2 is a2
        # dj/dw2 can be computed by doing a transposed dotted with delta3
        djdw2 = np.dot(self.a2.T, delta3)

        # dj/dw1 = -(y-yHat)(dyHat/dz3)(dz3/dw1)
        #        = -(y-yHat)f'(z3)(dz3/dw1)
        #        = delta3(dz3/dw1)
        #        = delta3(dz3/da2)(da2/dw1)
        #        = delta3*w2transposed*(da2/dz2)(dz2/dw1)
        #        = delta3*w2transposed*f'(z2)(dz2/dw1)
        # delta2 = delta3*w2transposed*f'(z2)
        delta2 = np.dot(delta3, self.w2.T)*self.sigmoidPrime(self.z2)
        # dz2/dw1 is x
        # djdw1 can be computed by doing a transposed dotted with delta2
        djdw1 = np.dot(x.t, delta2)

        return djdw1, djdw2

    def sigmoid(self, z):
        # apply sigmoid activation function
        return 1/(1+np.exp(-z))

    def sigmoidPrime(z):
        # derivate of sigmoid function
        return np.exp(-z)/((1+np.exp(-x))**2)

NN = Neural_Network()
yHat = NN.forward(x)
print yHat
print y
