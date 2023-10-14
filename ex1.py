import math

import numpy as np
import scipy.optimize
import sklearn.linear_model
from sklearn import linear_model
import matplotlib.pyplot as plt

class PoissonModel():
    def __init__(self):
        self.λ_ = np.nan

    def logLik(self, x, θ):
        λ = θ[0]
        if λ == 0:
            if np.sum(x == 0) == len(x):
                return 0
            else:
                return -math.inf
        return np.sum(np.multiply(x,np.log(λ)) - λ - np.log(scipy.special.factorial(x)))


    def fit(self, x):
        initial_guess = [1]
        (self.λ_,) = np.exp(scipy.optimize.fmin(lambda λ : -self.logLik(x,np.exp(λ)), initial_guess))



p = PoissonModel()
arr = [0,0,0]
p.fit(arr)
print(p.λ_)