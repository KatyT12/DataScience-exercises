import numpy as np
import scipy
import matplotlib.pyplot as plt


x = np.array([4.3, 2.8, 3.9, 4.1, 9, 4.5, 3.3])

num_samples = 300000
μ_samples = np.random.normal(loc = 0, scale = 5, size = num_samples)
choice_samples = np.random.choice([0,1],p=[0.99,0.01], size=num_samples)

μ_values = scipy.stats.norm.pdf(x[:,None], loc=μ_samples, scale = 0.5)
c = scipy.stats.cauchy.pdf(x)
cauchy_values = np.full(shape=(len(x), num_samples),fill_value = np.array(c)[:,None])

weights = (cauchy_values *0.01) + (μ_values*0.99)
weights[4] = np.where(choice_samples == 1, cauchy_values[4], μ_values[4])
weights = np.prod(weights, axis=0)
weights = weights/np.sum(weights)

fig, ax = plt.subplots()

print(np.sum(choice_samples * weights)) # Prints 0.9999999999999993

ax.hist(μ_samples,weights=weights,density=True,bins = 200)
plt.show()
