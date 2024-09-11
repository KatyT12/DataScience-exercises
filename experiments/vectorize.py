import numpy as np
# first excersise
def rgalazies(size, prob = [0.28, 0.54, 0.18], μ = [9740, 21300, 15000], σ = [340, 1700, 10600]):
    res = []
    for _ in range(size()):
        cluster = np.random.choice([0,1,2], p=prob)
        μi, σi = np.random.normal(loc = μi, scale = σi)
        res.append(np.random.normal(loc = μi, scale = σi))
    return res

# Vectorize it
def vgalazies(size, prob = [0.28, 0.54, 0.18], μ = [9740, 21300, 15000], σ = [340, 1700, 10600]):
    res = np.random.choice([0,1,2], p = prob, size = size)
    return np.random.normal(loc = μ[res], scale = σ[res])

""" Actual answer
def rgalaxies(size, p, μ, σ):
cluster = np.random.choice([0,1,2], p=p, size=size)
return np.random.normal(loc=np.array(μ)[cluster], scale=np.array(σ)[cluster])
"""

# Second exercise
def compute(x, p, μ, σ):
    res = np.zeros(len(x))
    for pk, μk, σk in zip(p, μ, σ):
        res += pk * 1/(2 * np.pi * σk**2) * np.exp(-((x - μk)**2)/ (2 * σk**2))
    return np.sum(np.log(sum))
