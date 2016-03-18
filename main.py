import numpy as np
from dataLoader import Data
from neuralNet import Neural_Network
from trainer import Trainer
from helpers import computeNumericalGradient

D = Data()
data = D.readData('cutData.csv')

# 70% of the data is used to train the neural network
trainData, testData = D.splitData(data, 0.7)

trainX, trainY = D.loadData(trainData)
trainX, trainY = D.toMatrix(trainX, trainY)

testX, testY = D.loadData(testData)
testX, testY = D.toMatrix(testX, testY)

NN = Neural_Network()
T = Trainer(NN)
T.train(trainX, trainY)
result = NN.forward(testX)

# used to test gradient descent
# if this number is too big, then the result will be garbage
numgrad = computeNumericalGradient(NN, trainX, trainY)
grad = NN.computeGradients(trainX, trainY)
print np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)

print 'real data'
print testY

print 'result'
print result
