import numpy as np
from scipy import optimize

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

class trainer(object):
    def __init__(self, N):
        # make local reference to Neural Network
        self.N = N

    def costFunctionWrapper(self, params, x, y):
        self.N.setParams(params)
        cost = self.N.costFunction(x, y)
        grad = self.N.computeGradients(x, y)
        return cost, grad

    def callBackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.x, self.y))

    def train(self, x, y):
        # make internal variable for callback function
        self.x = x
        self.y = y

        # make empty list to store costs
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}

        # BFGS (Broyden–Fletcher–Goldfarb–Shanno) approximates Newton's method
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = 'BFGS', args=(x, y), options=options, callback=self.callBackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


def computeNumericalGradient(N, x, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for i in range(len(paramsInitial)):
        # set pertubation vector
        perturb[i] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(x, y)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(x, y)

        # compute numerical gradient
        numgrad[i] = (loss2 - loss1) / (2*e)

        # return the value we changed back to zero
        perturb[i] = 0

    # return params to original value
    N.setParams(paramsInitial)

    return numgrad


NN = Neural_Network()
T = trainer(NN)
T.train(x, y)
print NN.forward(x)
