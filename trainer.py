import numpy as np
from scipy import optimize

class Trainer(object):
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

        # BFGS (Broyden-Fletcher-Goldfarb-Shanno) approximates Newton's method
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = 'BFGS', args=(x, y), options=options, callback=self.callBackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
