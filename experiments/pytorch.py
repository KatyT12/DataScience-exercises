import math

import pandas
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt

url = "https://www.cl.cam.ac.uk/teaching/current/DataSci/data/xkcd.csv"
xkcd = pandas.read_csv(url)


fig, ax = plt.subplots(figsize=(4,3))

#plt.show()


# Fitting with scipy
def μ(x, a,b,c):
    return a + b*x + c*x**2
def logPr(y,x,θ):
    a, b, c, σ = θ
    return np.sum(scipy.stats.norm.logpdf(y, loc = μ(x,a,b,c), scale = np.sqrt(σ**2)))

initial_guess = [1,1,0,1]
ahat, bhat, chat, σhat = scipy.optimize.fmin(lambda θ : -logPr(xkcd.y, xkcd.x, [θ[0], θ[1], θ[2], math.exp(θ[3])]), initial_guess)
σhat = math.exp(σhat)

xnew = np.linspace(0,10,100)
ynew = μ(xnew, ahat, bhat, chat)
ax.plot(xnew, ynew, ahat, bhat, chat)
ax.scatter(xkcd.x, xkcd.y)
ax.plot(xnew, ynew, color='black')
ax.fill_between(xnew, ynew - 2  * σhat,ynew + 2*σhat, color = 'steelblue', alpha=0.6)

plt.show()