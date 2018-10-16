import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(-3.0, 3.0, 0.1)
y = (- 4.12414349 - 0.48007329 * x) / (-0.6168482)
ax.plot(x, y, c="blue")
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

