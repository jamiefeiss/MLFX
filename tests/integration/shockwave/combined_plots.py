import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

# multiple h5 files?
f = h5py.File('shockwave.h5', 'r')

dset1 = f['1'] # density
P = dset1['density'][...]
psi_re = dset1['psi_re']
psi_im = dset1['psi_im']
x = dset1['x'][...]
t = dset1['t'][...]

dset2 = f['2'] # fourier
k_density_re = dset2['k_density_re'][...]
k_density_im = dset2['k_density_im'][...]
kx = dset2['kx'][...]
tk = dset2['t'][...]

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

# dset3 = f['3'] # fisher info
# tf = dset3['t'][...]
# F_Q_1 = dset3['F_Q_1'][...]
# F_Q_2_re = dset3['F_Q_2_re'][...]
# F_Q_2_im = dset3['F_Q_2_im'][...]
# F_C = dset3['F_C'][...]
# F_Q_2 = F_Q_2_re + 1j * F_Q_2_im
# F_Q = N * 4 * (F_Q_1 - np.abs(F_Q_2)**2)
# F_S = (2 * math.pi * R**2)**2

dset5 = f['5'] # functions
psi_init = dset5['psi_init'][...]
laser = dset5['laser'][...]
x_3 = dset5['x'][...]

# samples
imag_samples = 0
evo_samples = 1000
t_imag_start = 0
t_imag_end = t_imag_start + imag_samples
t_evo_start = 0
t_evo_end = t_evo_start + evo_samples - 1

# functions
def plot(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([-1, 1])
    ax.set_title(title)
    fig.savefig(filename + '.png')

def density_time_slice(time_index, title, filename):
    fig, ax = plt.subplots()
    ax.plot(x/(L/2), P[time_index])
    ax.set_xlabel(r'$\frac{x}{\pi}$')
    ax.set_ylabel(r'$|\psi|^2$')
    ax.set_ylim([0, 1])
    ax.set_title('Density at {} (t={})'.format(title, str(round(t[time_index], 4))))
    fig.savefig(filename + '.png')

# plot gaussian laser
fig, ax = plt.subplots()
ax.plot(x_3/(L/2), laser)
# ax.set_xlabel(r'$\frac{x}{\pi}$')
ax.set_xlabel(r'$x/\pi$')
ax.set_ylabel('V')
ax.set_ylim([-0.5, 0.1])
ax.set_title('Gaussian potential')
fig.savefig('gaussian' + '.png')

# plot ground state density
fig, ax = plt.subplots()
ax.plot(x/(L/2), P[t_imag_end])
ax.set_xlabel(r'$\frac{x}{\pi}$')
ax.set_ylabel(r'$|\psi|^2$')
ax.set_ylim([0, 0.3])
ax.set_title('Ground state density')
fig.savefig('ground_state' + '.png')

# plot shockwave propagation
fig, axes = plt.subplots(3, sharex=True, sharey=True)
# fig.suptitle('Shockwave propagation (R={}, A={}, w={})'.format(int(R), A, w))
fig.suptitle('Shockwave propagation')
axes[0].plot(x/(L/2), P[0], label='t={}'.format(round(t[0] - T_imag, 4)))
axes[1].plot(x/(L/2), P[400], label='t={}'.format(round(t[400] - T_imag, 4)))
axes[2].plot(x/(L/2), P[800], label='t={}'.format(round(t[800] - T_imag, 4)))
for ax in axes:
    # ax.set(xlabel=r'$\frac{x}{\pi}$', ylabel=r'$|\psi|^2$', ylim=[0, 0.3])
    ax.legend(handlelength=0)
# for ax in axes:
#     ax.label_outer()
plt.ylim([0, 0.3])
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.xlabel(r'$\frac{x}{\pi}$')
plt.xlabel(r'$x/\pi$')
plt.ylabel(r'$|\psi|^2$')

# ax.set_xlabel(r'$\frac{x}{\pi}$')
# ax.set_ylabel(r'$|\psi|^2$')
# ax.set_ylim([0, 0.3])
# ax.set_title('Final')
fig.savefig('final_density' + '.png')

# plot combined fourier density and plateaus for a single wavenumber for a single file

# plot multiple wavenumber plateaus for a single file

# 