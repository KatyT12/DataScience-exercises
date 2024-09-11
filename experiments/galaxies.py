
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math

# Questions
# Why use log? why not just use the normal formular? numbers too big?

url = 'https://www.cl.cam.ac.uk/teaching/2223/DataSci/data/galaxies_orig.csv'
galaxies = pandas.read_csv(url)
galaxies = galaxies['velocity'].values
ϕ = scipy.stats.norm.pdf

def transform_par(θ,n):
    q = [*θ[0:n-1],0]
    μ = θ[n-1: 2 * n-1]
    τ = θ[2 * n -1 : ]
    p = np.exp(q)
    p = p / np.sum(p)
    σ = np.exp(τ)
    return p, μ, σ

def logPr(x, θ, n):
    p, μ, σ = transform_par(θ,n)
    lik = 0 # Don't know how to vectorize it
    for pi, μi, σi in zip(p, μ, σ):
        lik += pi * ϕ(x, loc=μi, scale=σi)
    return np.log(lik) # Sum afterwards because we also want to use this to draw the graph

initial_guess = [0, 0, 10000, 20000, 24000, math.log(1000), math.log(5000), math.log(8000)]
θ = scipy.optimize.fmin(lambda theta : -np.sum(logPr(galaxies, theta, 3)), initial_guess, maxiter=5000)
# Should be θ with hat

fig, ax = plt.subplots(figsize=(6,3))
x = np.linspace(0, 40000, 200)
f = np.exp(logPr(x, θ,3))
ax.plot(x, f)
ax.hist(galaxies, edgecolor = "black",bins = np.linspace(0,40000,80), density=True,color = "green")
plt.show()