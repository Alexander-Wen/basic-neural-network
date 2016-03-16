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
data = D.readData('cutData.csv')

trainData, testData = D.splitData(data, 0.7)

trainX, trainY = D.loadData(trainData)
trainX, trainY = D.toMatrix(trainX, trainY)

testX, testY = D.loadData(testData)
testX, testY = D.toMatrix(testX, testY)

print 'real data'
print testY

NN = Neural_Network()
T = Trainer(NN)
T.train(trainX, trainY)
result = NN.forward(testX)

print result
