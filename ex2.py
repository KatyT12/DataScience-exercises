import math

import numpy as np
import scipy.optimize
import sklearn.linear_model
from sklearn import linear_model
import matplotlib.pyplot as plt


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