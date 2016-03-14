import numpy as np

class Neural_Network(object):
    def __init__(self):
        # Define Hyperparamters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 5

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
        djdw1 = np.dot(x.T, delta2)

        return djdw1, djdw2

    def sigmoid(self, z):
        # apply sigmoid activation function
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        # derivate of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)

    # helper functions
    def getParams(self):
        # get w1 and w2
        # ravel flattens arrays
        params = np.concatenate((self.w1.ravel(), self.w2.ravel()))
        return params

    def setParams(self, params):
        # set w1 and w2 using single parameter vector:
        w1_start = 0
        w1_end = self.hiddenLayerSize*self.inputLayerSize
        # reshape reshapes array into matrix
        self.w1 = np.reshape(params[w1_start:w1_end], (self.inputLayerSize, self.hiddenLayerSize))
        w2_end = w1_end+self.hiddenLayerSize*self.hiddenLayerSize
        self.w2 = np.reshape(params[w1_end:w2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, x, y):
        djdw1, djdw2 = self.costFunctionPrime(x, y)
        return np.concatenate((djdw1.ravel(), djdw2.ravel()))
