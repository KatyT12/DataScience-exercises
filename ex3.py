import math

import numpy as np
import scipy.optimize
import sklearn.linear_model
from sklearn import linear_model
import matplotlib.pyplot as plt


class StepPeriodicModel():
    def __init__(self):
        self.mindec = np.nan
        self.maxdec = np.nan

    def fit(self, t, temp):
        # data is from 1850s to 2020s, 40s = 0, 20s = 8
        self.mindec = math.floor(min(t)/10)*10
        self.maxdec = math.floor(max(t) / 10) * 10

        yr = [np.where(np.floor(t/10)*10 == year*10 + self.mindec, 1, 0) for year in range(int((self.maxdec - self.mindec)/10) + 1)]

        X = np.column_stack([np.sin(2 * np.pi * np.mod(t,1)), np.cos(2 * np.mod(t,1)), *yr])
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, temp)
        self.α, (self.β1, self.β2, self.γ) = model.intercept_, (model.coef_[0],model.coef_[1],[*model.coef_[2:],np.nan])
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
        return self.α + self.β1 * np.sin(2 * np.pi * np.mod(t,1)) + self.β2 * np.cos(2 * np.pi * np.mod(t,1)) + gamma_part