import numpy as np

class Score(object):
    def score(self, a, b):
        return sum((a.ravel() - b.ravel()) ** 2)
