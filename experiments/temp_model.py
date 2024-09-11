import numpy as np
import matplotlib as plt
import pandas
import sklearn.linear_model
from sklearn import *

climate = pandas.read_csv('https://www.cl.cam.ac.uk/teaching/2223/DataSci/data/climate.csv')
df = climate.loc[(climate.station == 'Cambridge') & (climate.yyyy >= 1985)]
t = df.yyyy + (df.mm-1)/12 # Predictor, feature
temp = (df.tmin + df.tmax)/2 # Labels

# Do model
X = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
model = sklearn.linear_model.LinearRegression()
model.fit(X,temp)
alpha, (beta1, beta2) = model.intercept_, model.coef_

ε = temp - model.predict(X)

