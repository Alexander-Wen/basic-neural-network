import numpy as np
import random

class Data(object):
    def readData(self, filePath):
        f = open(filePath)
        data = f.read()
        parsedData = data.split('\n')
        parsedData.pop(0)
        a = []
        for i in parsedData:
            if (not(i)):
                continue
            a.append(i)

        return a

    def loadData(self, data):
        x = []
        y = []
        # TODO: make this a map (functional)
        for i in data:
            a, b = self.prepareData(i)
            x.append(a)
            y.append(b)

        return x, y

    def toMatrix(self, x, y):
        xx = np.array(x, dtype=float)
        yy = np.array(y, dtype=float)

        # scale inputs so they are the same format
        xx = xx/np.amax(xx, axis=0)
        yy = yy/100

        return xx, yy


    def prepareData(self, data):
        cleanData = data.split(',');
        return [cleanData[0], cleanData[1]], [cleanData[2]]

    def splitData(self, data, percentage):
        a = []
        b = []
        for i in data:
            if (random.randint(1,100) < percentage*100):
                a.append(i)
            else:
                b.append(i)
        return a, b
