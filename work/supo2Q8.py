import numpy as np
import scipy
import matplotlib.pyplot as plt


x = np.array([4.3, 2.8, 3.9, 4.1, 9, 4.5, 3.3])

num_samples = 100000
μ_samples = np.random.normal(loc = 0, scale = 0.5, size = num_samples)
# Matrix of choices for each x
choice_samples = np.random.choice([1,0],p=[0.99,0.01],size=(len(x), num_samples))


x_μ = scipy.stats.norm.pdf(x[:,None], loc=μ_samples, scale = 0.5)

# Matrix of cauchy values, each column is the same but matrix for convenience of next lines
c = scipy.stats.cauchy.pdf(x)
cauchy_values = np.full(shape=(len(x), num_samples),fill_value = np.array(c)[:,None])

weights = np.sum(np.where(choice_samples, x_μ , cauchy_values),axis=0)
weights = weights/np.sum(weights)

#Plot posterior distrobution
fig, ax = plt.subplots()
print(choice_samples[4])
ax.hist(choice_samples[4],weights=weights,density=True,bins = 80)
plt.show()
