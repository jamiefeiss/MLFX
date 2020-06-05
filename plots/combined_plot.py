import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 20, 100)
v1 = 0.5*(x)**2
v2 = 0.5*(x-10)**2

t = np.linspace(0, 1, 1000)
l1 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-1)))
l2 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-2)))
l3 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-3)))
l4 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-4)))
l5 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-5)))
l10 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-10)))
l50 = 1 - (1/(1+((t+0.0001)**(-1)-1)**(-50)))

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(x, v1, 'r-', label='t=0')
ax1.plot(x, v2, 'b-', label='t=T')
ax1.set_xlim([-5, 20])
ax1.set_ylim([0, 10])
ax1.set_xticks([0, 10])
ax1.set_xticklabels(['0', '$x_0$'])
ax1.set_xlabel('x')
ax1.set_ylabel('V')
ax1.set_title('Potential function')
ax1.legend()

ax2.plot(t, l1, 'r-', label='k=1')
ax2.plot(t, l2, 'y-', label='k=2')
ax2.plot(t, l3, 'g-', label='k=3')
ax2.plot(t, l4, 'c-', label='k=4')
ax2.plot(t, l5, 'b-', label='k=5')
ax2.plot(t, l10, 'm-', label='k=10')
ax2.plot(t, l50, 'k-', label='k=50')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['0', '1'])
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['0', '1'])
ax2.set_xlabel('t')
ax2.set_ylabel('$\lambda(t)$')
ax2.set_title('Timing function at different values of k')
ax2.legend()

fig.savefig('combined_plot.png')