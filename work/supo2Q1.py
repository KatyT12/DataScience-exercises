import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Outline, left eye, right eye, smile
def rxy(s = 1):
    eye_var = 0.07
    smile_var = 0.06
    outline_var = 0.05

    k = np.random.choice([0, 1, 2, 3],size=s,p = [0.6, 0.1, 0.1, 0.2])

    theta_outline = np.random.uniform(low=0, high=2*math.pi, size=s)
    theta_smile = np.random.uniform(low=math.pi, high=2*math.pi, size=s)

    x = np.array([
        np.random.normal(loc=np.cos(theta_outline), scale=outline_var, size=s),
        np.random.normal(loc=-0.3, scale=eye_var, size=s),
        np.random.normal(loc=0.3, scale=eye_var,size=s),
        np.random.normal(loc=0.5 * np.cos(theta_smile), scale=smile_var,size=s)
    ])

    y = np.array([
        np.random.normal(loc=np.sin(theta_outline), scale=outline_var,size=s),
        np.random.normal(loc=0.4, scale=eye_var,size=s),
        np.random.normal(loc=0.4, scale=eye_var,size=s),
        np.random.normal(loc=0.5 * np.sin(theta_smile), scale=smile_var,size=s)
        ]
    )

    return np.array([np.choose(k,x), np.choose(k, y)])

[x, y] = rxy(s = 10000)

# Plots
gs = gridspec.GridSpec(2,2)
ax = plt.subplot(gs[1,0])
ax.scatter(x,y,s=0.5)

ax = plt.subplot(gs[0,0])
ax.hist(x,density=True,bins=50)

ax = plt.subplot(gs[1,1])
ax.hist(y,density=True,bins=50, orientation=u'horizontal')
plt.show()
""" Alternitive code for plotting the histrograms
fig,((ax_x,dummy),(ax_xy,ax_y)) = plt.subplots(2,2, figsize=(4,4),
sharex='col', sharey='row', gridspec_kw='height_ratios':[1,2], 'width_ratios':[2,1])
dummy.remove()
ax_xy.scatter(xy[:,0], xy[:,1], s=3, alpha=.1)
ax_x.hist(???, density=True, bins=60) # fill in the ???
ax_y.hist(???, density=True, bins=60, orientation='horizontal') # fill in the ???
plt.show()
"""

nsamples = 300000
def get_interval(p, weights, samples):
    val = (1 - p) / 2
    i = np.argsort(samples)
    theta, F = samples[i], np.cumsum(weights[i])
    lo = theta[F < val][-1]
    hi = theta[F > p + val][0]
    return (lo, hi)

def exp_poisson_confint(p, x):
    # Input: p=0.95 and x=np.array([4,0,0,0,3,1,1,0,1,1,1,0,1,0,1,2,3,0,1,1])
    # TODO: compute a p-confidence interval for Θ
    lambda_samples = np.random.exponential(size=nsamples, scale = 1)

    # Do it in a better way
    w = scipy.stats.poisson.logpmf(x[:, None], lambda_samples)
    w = np.sum(w,axis=0)
    w = w - np.max(w)
    w = np.exp(w)/np.sum(np.exp(w))
    #fig, ax = plt.subplots()
    #ax.hist(lambda_samples,weights=w, density=True, bins = np.linspace(0,5,60))
    #plt.show()

    # Find low adnd high
    return get_interval(p, w, lambda_samples)


def exp_uniform_confint(p, x, λ0, μ0):
    # Input: p=0.95, λ0=0.5, μ0=1.0, x=np.array([2, 3, 2.1, 2.4, 3.14, 1.8])
    # TODO: compute a p-confidence interval for B
    asamp = np.random.exponential(scale= 1/λ0, size=nsamples)
    bsamp = np.random.exponential(scale=1/μ0, size=nsamples)

    #Do we need to use log?
    w = np.where((np.min(x) >= asamp) & (np.max(x) <= bsamp + asamp), 1, 0)

    weights = np.power(1/(bsamp),len(x)) * w
    weights = weights/np.sum(weights)
    return get_interval(p, weights, bsamp)

