import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn import *

iris = pandas.read_csv("https://www.cl.cam.ac.uk/teaching/2223/DataSci/data/iris.csv")
one, x = np.ones(len(iris)), iris["Sepal.Length"]
y = iris["Petal.Length"]
model = linear_model.LinearRegression(fit_intercept = False)
model.fit(np.column_stack([one, x, x**2]), y)
alpha, beta, gamma = model.coef_

model2 = sklearn.linear_model.LinearRegression()
model2.fit(np.column_stack([x, x**2]),y)
alpha, (beta, gamma) = model2.intercept_, model2.coef_

newx = np.linspace(4.2, 8.2, 20)
ones = np.ones(20);
predy = model2.predict(np.column_stack([newx,newx**2]))

fig,ax = plt.subplots(figsize = (4.5,3))
ax.plot(newx, predy, color='0.5', zorder=-1, linewidth = 1, linestyle='dashed')
ax.scatter(iris['Sepal.Length'], iris['Petal.Length'], alpha=.3)
ax.set_ylim(0,7.5)
ax.set_ylabel('Petal.Length')
ax.set_xlabel('Sepal.Length')
plt.show()