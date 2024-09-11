import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def f (σ) :
    return np.exp(-3 * 0.5/np. power(σ ,2)) / np.sqrt (2*np. pi*np. power(σ ,2))

fig, ax = plt.subplots()
σ = np.linspace(0,10,100)[1:] # Remove σ = 0 where the function doesn't work
ax.plot(σ, f(σ))
plt.show()

(T, ) = scipy.optimize.fmin(lambda t : -f(np.exp(t)), np.log(2))
sigma = np.exp(T)



def g(theta):
    p1, p2, p3 = theta.T
    
