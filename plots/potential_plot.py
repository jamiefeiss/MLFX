import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 20, 100)
v1 = 0.5*(x)**2
v2 = 0.5*(x-10)**2

fig, ax = plt.subplots()
ax.plot(x, v1, 'r-', label='t=0')
ax.plot(x, v2, 'b-', label='t=T')
ax.set_xlim([-5, 20])
ax.set_ylim([0, 10])
ax.set_xticks([0, 10])
ax.set_xticklabels(['0', '$x_0$'])
ax.set_xlabel('x')
ax.set_ylabel('V')
ax.set_title('Potential function')
ax.legend()
fig.savefig('potential_plot.png')
