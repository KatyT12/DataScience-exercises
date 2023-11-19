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


class PiecewiseLinearModel():
    # Params
    def __init__(self):
        self.α = np.nan
        self.β1 = np.nan
        self.β2 = np.nan
        self.inflection_x = np.nan

    def fit(self, x, y, inflection_x):
        self.inflection_x = inflection_x
        # i1 = 1 if(x < inflection_x)
        t1 = np.where(x <= inflection_x,1,0)

        X = np.column_stack([t1*x + (1-t1)*inflection_x, (1-t1)*(x - inflection_x) ])
        model = sklearn.linear_model.LinearRegression()
        model.fit(X,y)
        self.α, (self.β1,self.β2) = model.intercept_, model.coef_

    def predict(self,x):
            t1 = np.where(x <= self.inflection_x,1,0)
            val = self.α + self.β1 * (t1*x + (1-t1)*self.inflection_x) + self.β2*(1-t1)*(x - self.inflection_x)
            return val


class StepPeriodicModel():
    def __init__(self):
        self.mindec = np.nan
        self.maxdec = np.nan

    def fit(self, t, temp):
        # get min and max decaes
        self.mindec = math.floor(min(t)/10)*10
        self.maxdec = math.floor(max(t) / 10) * 10
        # One hot coding years for each time
        yr = [np.where(np.floor(t/10)*10 == year*10 + self.mindec, 1, 0) for year in range(int((self.maxdec - self.mindec)/10) + 1)]
        #Set up linear model
        X = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), *yr])
        #Fit
        model = sklearn.linear_model.LinearRegression(fit_intercept = False)
        model.fit(X, temp)

        (self.β1, self.β2, self.γ) = (model.coef_[0],model.coef_[1],[*model.coef_[2:],np.nan])
        self.γ = np.array(self.γ)


    def predict_step(self, t):
        print(t)
        l = (np.floor(t/10)*10-self.mindec)/10
        l = l.astype(int)
        replace_mask = np.where(np.logical_or(l< 0, l >= len(self.γ)-1))
        l[replace_mask] = len(self.γ)-1
        print(l)
        gamma_part = np.take(self.γ,l)
        print(self.γ)
        print(self.β1, self.β2)
        print(gamma_part)
        return gamma_part





p = PoissonModel()
arr = [0,0,0]
p.fit(arr)
print(p.λ_)