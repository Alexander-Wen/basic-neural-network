import numpy as np
from dataLoader import Data
from neuralNet import Neural_Network
from trainer import Trainer

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

D = Data()
a = D.readData('cutData.csv')
x, y = D.loadData(a)

print 'real data'
print y

NN = Neural_Network()
T = Trainer(NN)
T.train(x, y)
result = NN.forward(x)

print result
