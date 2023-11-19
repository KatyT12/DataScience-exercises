import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas

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

# A + bsin(2pi(t + phi)) + C(t-2000) + N(0,sigma^2)
#A normal, B normal


def get_weights(t,temp, climatemodel,asamp, csamp):
    a, b, φ, c, σ, σa, σc = climatemodel
    w = np.zeros(len(asamp))
    for time, temperature in zip(t,temp):
        μ = (time - 2000) * csamp + asamp + (b * np.sin(2 * np.pi * (time + φ)))
        w += scipy.stats.norm.logpdf(temperature, loc=μ, scale=σ)
    w = w - np.max(w)
    w = np.exp(w) / np.sum(np.exp(w))
    return w



def climate_inc_confint(p, t, temp, climatemodel):
    # Input: p=0.95, t and temp taken from the dataset, climatemodel an object
    # with attributes (a,b,φ,c,σ,σa,σb) specifying the model parameters.

    # Have to do this for the sake of memory
    nsamples = 100000
    a, b, φ, c, σ, σa, σc = climatemodel
    # TODO: compute a p-confidence interval for C
    asamp = np.random.normal(loc=a, scale=σa, size =nsamples)
    csamp = np.random.normal(loc=c, scale=σc, size=nsamples)

    # Get weights
    """
    μ = ( (t-2000))[:, None]* csamp + asamp + (b * np.sin(2 * np.pi * (t + φ)))[:,None]

    w = scipy.stats.norm.logpdf(temp[:, None],loc= μ, scale= σ)
    w = np.sum(w, axis=0) # Likelihood of dataset for each sample
    w = w - np.max(w)
    w = np.exp(w) / np.sum(np.exp(w))
    """
    w = get_weights(t,temp,climatemodel, asamp,csamp)

    return get_interval(p,w,csamp)


"""
You were thinking about it in the wrong way here, we have no extra uncertainty, it is purely in the 
parameters.
"""

def climate_pred_confint(p, newt, t, temp, climatemodel):
    nsamples = 10000
    a, b, φ, c, σ, σa, σc = climatemodel

    asamp = np.random.normal(loc=a, scale=σa, size=nsamples)
    csamp = np.random.normal(loc=c, scale=σc, size=nsamples)

    # Get weights for each sample
    w = get_weights(t,temp, climatemodel, asamp, csamp)

    # expected cdf of Pr(Temp | data) =  E(Pr(Temp < hi | params) | data), this
    # is instead of aproximating cdf Pr(Temp | data) from the average
    tempSamples = np.linspace(8,25, 5000)

    μ = asamp + (newt-2000) * csamp
    expected_cdf = np.array([np.average(scipy.stats.norm.cdf(tem, loc=μ, scale = σ), weights=w) for tem in tempSamples])

    #expected_cdf = np.average(
    #scipy.stats.norm.cdf(tempSamples[:,None], loc = μ[None,:], scale = σ), weights=w,axis=1)

    val = (1 - p) / 2
    (lo, hi) = np.interp([val, p + val], expected_cdf, tempSamples)



    """
    fig, ax = plt.subplots()

    
    ax.axhline(val)
    ax.axhline(val+p)
    ax.axvline(lo)
    ax.axvline(hi)
    ax.plot(tempSamples, expected_cdf)

    plt.show()
    """

    return (lo,hi)




# TEST


#exp_uniform_confint(0.95,λ0=0.5, μ0=1.0, x=np.array([2, 3, 2.1, 2.4, 3.14, 1.8]))
url = 'https://www.cl.cam.ac.uk/teaching/current/DataSci/data/climate_202309.csv'
climate = pandas.read_csv(url)
df = climate.loc[(climate.station=='Cambridge') & (climate.yyyy>=2010)]
t,temp = df.t.values, df.temp.values

import collections
ClimateModel = collections.namedtuple('ClimateModel', ['a','b','φ','c','σ','σa','σc'])
climatemodel = ClimateModel(a=10, b=6.6826, φ=-0.27731, c=0, σ=1.4183, σa=5, σc=0.1)

#print(climate_inc_confint(0.95, t=t, temp=temp, climatemodel=climatemodel))
print(climate_pred_confint(0.95, 2050, t=t, temp=temp, climatemodel=climatemodel))