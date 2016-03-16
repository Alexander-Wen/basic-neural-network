import numpy as np
import re

class Data(object):
    def readData(self, filePath):
        f = open(filePath)
        data = f.read()
        parsedData = data.split('\n')
        parsedData.pop(0)
        return parsedData

    def loadData(self, data):
        x = []
        y = []
        # TODO: make this a map (functional)
        for i in data:
            if (not(i)):
                continue
            a, b = self.prepareData(i)
            x.append(a)
            y.append(b)

        xx = np.array(x, dtype=float)
        yy = np.array(y, dtype=float)

        # scale inputs so they are the same format
        xx = xx/np.amax(xx, axis=0)
        yy = yy/100

        return xx, yy

    def prepareData(self, data):
        cleanData = data.split(',');
        return [cleanData[0], cleanData[1]], [cleanData[2]]
