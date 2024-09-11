import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt

n = 400

k = np.random.choice(4, p=[.6,.3,.05,.05], size=n)
t = np.random.uniform(size=n)
x = np.column_stack([np.sin(2*np.pi*t), 0.55*np.sin(2*np.pi*(0.4*t+0.3)), -0.3*np.ones(n), 0.3*np.ones(n)])
y = np.column_stack([np.cos(2*np.pi*t), 0.55*np.cos(2*np.pi*(0.4*t+0.3)), 0.3*np.ones(n), 0.3*np.ones(n)])
xy = np.column_stack([x[np.arange(n), k], y[np.arange(n), k]])
xy = np.random.normal(loc=xy, scale=.08)



samp_size = 1000

t_samp = np.random.uniform(size=samp_size)
k_samp = np.random.choice(4, p=[.6,.3,.05,.05], size=samp_size)

# Used later
locx = [np.sin(2*np.pi*t_samp), 0.55*np.sin(2*np.pi*(0.4*t_samp+0.3)), -0.3*np.ones(samp_size), 0.3*np.ones(samp_size)]
locy = [np.cos(2*np.pi*t_samp), 0.55*np.cos(2*np.pi*(0.4*t_samp+0.3)), 0.3*np.ones(samp_size), 0.3*np.ones(samp_size)]

xs = np.random.normal(loc=np.column_stack(locx), scale=.08)
x_samp = xs[np.arange(samp_size),k_samp]
y_vals = np.full(shape=(samp_size), fill_value=0.3)


w = np.sum(np.multiply( 
            np.transpose(scipy.stats.norm.pdf(x_samp, loc=locx, scale = 0.08) * 
            scipy.stats.norm.pdf(y_vals, loc=locy, scale = 0.08)), [.6,.3,.05,.05])
              , axis=1)/ samp_size

plt.hist(x_samp, weights=w, density=True)
plt.show()