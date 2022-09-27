from random import sample
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 1)

x = np.arange (0.01, 1, 0.01)
y = beta.pdf(x, 4, 5)#**20 * beta.pdf(x, 2, 3)**30
# y2 = beta.cdf(x, 22, 32)#**20 * beta.cdf(x, 2, 3)**30

ax.plot(x, y, lw=3, label="beta")
# ax.plot(x, y2, lw=3, label="beta cdf")
ax.legend(loc='best', frameon=False)
plt.xlim([0, 1])
plt.ylim([0, 8]) #1000000000
plt.xlabel("Theta")
plt.ylabel("P(Theta|X)")
plt.show()