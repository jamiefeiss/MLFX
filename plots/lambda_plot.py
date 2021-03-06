import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 1, 1000)
l1 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-1)))
l2 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-2)))
l3 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-3)))
l4 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-4)))
l5 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-5)))
l10 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-10)))
l50 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-50)))

fig, ax = plt.subplots()
ax.plot(t, l1, 'r-', label='k=1')
ax.plot(t, l2, 'y-', label='k=2')
ax.plot(t, l3, 'g-', label='k=3')
ax.plot(t, l4, 'c-', label='k=4')
ax.plot(t, l5, 'b-', label='k=5')
ax.plot(t, l10, 'm-', label='k=10')
ax.plot(t, l50, 'k-', label='k=50')
ax.set_xticks([0, 1])
ax.set_xticklabels(['0', '1'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['0', '1'])
ax.set_xlabel('t')
ax.set_ylabel('$\lambda(t)$')
ax.set_title('Timing function at different values of k')
ax.legend()
fig.savefig('lambda_plot.png')
