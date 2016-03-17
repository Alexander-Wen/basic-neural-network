import numpy as np
from dataLoader import Data
from neuralNet import Neural_Network
from trainer import Trainer

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
