import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Range 0-100 y
def get_rand_points(num):
    arr = np.random.randint(0,100,num)
    return arr

# Bezier curve
def get_bezier_points(points, num):
    n = len(points) - 1
    bez_points = np.arange(num+1)
    for i in range(num+1):
        t = i/num
        sum = 0
        for j in range(n+1):
            # Binomial
            val = pow(t,j) * pow(1-t, n-j) * math.comb(n, j) * points[j]
            sum += val
        print(sum)
        bez_points[i] = sum
    return bez_points




num = 6
num_bez = 600

p1 = get_rand_points(num)
p2 = get_bezier_points(p1,num_bez)

x_axis1 = np.linspace(0,100, num)
x_axis2 = np.linspace(0,100, num_bez+1)

fig, ax = plt.subplots()
ax.plot(x_axis2, p2)
ax.scatter(x_axis1, p1)

plt.show()
