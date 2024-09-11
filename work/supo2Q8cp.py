import numpy as np
import scipy
import matplotlib.pyplot as plt


x = np.array([4.3, 2.8, 3.9, 4.1, 9, 4.5, 3.3])

num_samples = 10000
μ_samples = np.random.normal(loc = 0, scale = 5, size = num_samples)


μ_values = scipy.stats.norm.pdf(x[:,None], loc=μ_samples, scale = 0.5) * 0.99
c = scipy.stats.cauchy.pdf(x) * 0.99
cauchy_values = np.full(shape=(len(x), num_samples),fill_value = np.array(c)[:,None]) * 0.01

weights = cauchy_values +μ_values
#weights[4] = np.where(choice_samples == 0, cauchy_values[4], μ_values[4])
weights = np.prod(weights, axis=0)
weights = weights/np.sum(weights)

fig, ax = plt.subplots()

ax.hist(μ_samples,weights=weights,density=True,bins = 300)
plt.show()