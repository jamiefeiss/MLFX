import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

f = h5py.File('shockwave.h5', 'r')

dset4 = f['4'] # constants
R = dset4['radius'][...]
Omega = dset4['omega'][...]
delta = dset4['d_omega'][...]
L = dset4['length'][...]
N = dset4['no_atoms'][...]
phi = dset4['phi'][...]
g = dset4['non_lin'][...]
A_psi = dset4['amplitude_psi'][...]
w_psi = dset4['width_psi'][...]
A = dset4['amplitude'][...]
w = dset4['width'][...]
T_imag = dset4['t_imag'][...]
T_evo = dset4['t_evo'][...]

dset1 = f['1'] # density
P = dset1['density'][...]
psi_re = dset1['psi_re']
psi_im = dset1['psi_im']
x = dset1['x'][...]
t = dset1['t'][...] - T_imag

# samples
imag_samples = 0
evo_samples = 1000
t_imag_start = 0
t_imag_end = t_imag_start + imag_samples
t_evo_start = 0
t_evo_end = t_evo_start + evo_samples - 1

thetas = -np.linspace(-np.pi/2, 3*np.pi/2, P[0].shape[0])
fig, ax = plt.subplots(subplot_kw={'polar': True})

plt.tick_params(labelsize=16)
ax.spines['polar'].set_visible(False)

ax.plot(thetas, P[0], 'r:', label='t={}'.format(round(t[0], 4)))
ax.plot(thetas, P[120], 'b--', label='t={}'.format(round(t[175], 4)))
ax.plot(thetas, P[350], 'k-', label='t={}'.format(round(t[350], 4)))
# ax.legend(loc=(0.9, 0.85), fontsize=13)
ax.grid(linewidth=0)
ax.set_theta_offset(np.pi/2)
ax.set_yticklabels([])
ax.set_xticklabels([])
# ax.set_xticks([0, np.pi/2, np.pi, 2.145*np.pi/2, 3*np.pi/2])
# ax.set_xticklabels(['0', r'$\frac{\pi R}{2}$', r'$\pi R$', r'$x_c$', r'$\frac{3\pi R}{2}$'])
radii = np.linspace(0, 0.4, P[0].shape[0] + 1).reshape(-1, 1)
radrange = radii.ptp()
ring_width = 0.4
ax.set_rlim(radrange * (1 - 1. / ring_width), radrange)
fig.savefig('ring_blank.png', dpi=300)