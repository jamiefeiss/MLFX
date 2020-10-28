import numpy as np
import matplotlib.pyplot as plt

with open('optimal.txt') as f:
    lines = f.readlines()
    x1 = [float(line.split()[0]) for line in lines]
    x2 = [float(line.split()[1]) for line in lines]

with open('cost.txt') as f:
    lines = f.readlines()
    y = [-float(line.split()[0]) for line in lines]

epochs = range(len(y))

# y_limit = 1.1 * max(y)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)
fig.set_size_inches(6.4, 3)
ax.plot(epochs, y)
# ax.plot(epochs, np.zeros(len(y)), color='gray', linestyle='dashed')
# ax.set_ylim([-y_limit, y_limit])
ax.set_xlabel('Epochs')
ax.set_ylabel(r'$|F_{2k}|$')
ax.set_title('Optimisation of Fourier magnitude')
fig.savefig('optimal_plot.png')