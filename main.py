import numpy as np
from trainer import Trainer
from neuralNet import Neural_Network

# input training data
# hours sleep, hours studying
x = np.array(([3,5], [5,1], [10,2]), dtype=float)
# result training data
# grade
y = np.array(([75],[82],[93]), dtype=float)
# scale inputs so they are in the same format
x = x/np.amax(x, axis=0)
y = y/100

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
T = Trainer(NN)
T.train(x, y)
print NN.forward(x)
