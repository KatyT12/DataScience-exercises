import numpy as np
import matplotlib.pyplot  as plt
import pandas

num = 50000
def sd_confint_parametric(p, x):
    # Input: p=0.95 and x = November temperatures in Cambridge from 2000 onwards
    # TODO: compute a p-confidence interval for σhat

    μhat = np.mean(x)
    def σhat(data):
        return np.sqrt(np.mean((np.mean(data) - data)**2))

    def rx_star():
        return np.random.normal(size=len(x), loc=μhat, scale=σhat(x))

    # Resampling
    σhat_ = np.array([σhat(rx_star()) for _ in range(num)])
    val = (1-p)/2
    lo, hi = np.quantile(σhat_, [val,p + val])
    return (lo, hi)

def sd_confint_nonparametric(p, x):
    def σhat(data):
        return np.sqrt(np.mean((np.mean(data) - data)**2))

    def rx_star():
        return np.random.choice(x, size=len(x))

    σhat_ = np.array([σhat(rx_star()) for _ in range(num)])
    val = (1-p)/2
    lo, hi = np.quantile(σhat_, [val,p + val])
    return (lo, hi)

def exp_equality_test(x, y):
    def λhat(data): return data.size/(np.sum(data))

    def test(x_,y_): return  λhat(y_) - λhat(x_)
    
    xy = np.append(np.array(x),np.array(y))
    λhatxy = λhat(xy)

    def rxystar():
        return (np.random.exponential(scale=1/λhatxy, size=len(x)),
                np.random.exponential(scale=1/λhatxy, size=len(y)))

    t_ = np.array([test(*rxystar()) for _ in range(num)])
    return 2*min(np.mean(t_ >= test(x,y)), np.mean(t_ <= test(x,y)))
