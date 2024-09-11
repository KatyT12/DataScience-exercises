import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import sklearn.linear_model


responses = pandas.read_csv("https://www.cl.cam.ac.uk/teaching/current/DataSci/data/responsetime_ms.txt", header=None)
responses = np.array(responses[0])
responses = np.sort(responses)
ecdf = np.arange(1,len(responses)+1)/len(responses)

etdf = np.log(1 - ecdf)
fig, ax = plt.subplots()
ax.plot(np.log(np.sort(responses)), etdf, drawstyle = 'steps-post')

ax.set_xlabel("Response times")
ax.set_ylabel("etdf")


# Fit line to graph
# etdf = m * responsees + c
line = np.log(responses[responses >= 23])
print(etdf[responses >= 23])
X = np.column_stack([line])
model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, etdf[responses >= 23])
c, m = model.coef_
ax.plot(np.log(responses), c + m*np.log(responses))

three_nines = (math.log(1-0.999)-c)/m
four_nines = (math.log(1-0.9999)-c)/m

three_nines = np.interp(x=[math.log(1-0.08)],xp=etdf, fp=np.sort(responses))

plt.show()



"""
gs = gridspec.GridSpec(1,3)

ax0 = plt.subplot(gs[0,0])
ax0.plot(np.sort(responses), ecdf, drawstyle = 'steps-post')
ax0.set_xlim([0,5])

ax1 = plt.subplot(gs[0,1])
ax1.plot(np.sort(responses), ecdf, drawstyle = 'steps-post')
ax1.set_xlim([0,100])

ax2 = plt.subplot(gs[0,2])
ax2.plot(np.sort(responses), ecdf, drawstyle = 'steps-post')

plt.show()
"""