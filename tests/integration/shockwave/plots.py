import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

f = h5py.File('shockwave.h5', 'r')

dset1 = f['1']
dset2 = f['2']
dset3 = f['3']

# density
P = dset1['density'][...]
x = dset1['x'][...]
t = dset1['t'][...]

# constants
R = dset2['radius'][...]
Omega = dset2['omega'][...]
L = dset2['length'][...]
g = dset2['non_lin'][...]
A_psi = dset2['amplitude_psi'][...]
w_psi = dset2['width_psi'][...]
A = dset2['amplitude'][...]
w = dset2['width'][...]
T_imag = dset2['t_imag'][...]
T_evo = dset2['t_evo'][...]

# functions
psi_init = dset3['psi_init'][...]
laser = dset3['laser'][...]
x_3 = dset3['x'][...]

# number of samples per integration block
imag_samples = 1000
evo_samples = 1000

# time indicies per integration block
t_imag_start = 0
t_imag_end = t_imag_start + imag_samples

t_evo_start = t_imag_end + 1
t_evo_end = t_evo_start + evo_samples - 1

# # density contour plot
# fig1, ax1 = plt.subplots()
# c1 = ax1.contourf(x/(L/2), t, P, cmap='Reds')
# ax1.set_xlabel(r'$\frac{x}{\pi}$')
# ax1.set_ylabel('t')
# ax1.set_title('Density, R={}'.format(R))
# fig1.colorbar(c1)
# fig1.savefig('density.png')

def plot(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, 1])
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

def polar_heat(time_index, title, filename):
    values = np.atleast_2d(P[time_index])
    thetas = np.linspace(-np.pi, np.pi, values.shape[1]).reshape(1, -1)
    radii = np.linspace(0, 1, values.shape[0] + 1).reshape(-1, 1)
    fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True})
    mesh = ax.pcolormesh(thetas, radii, values)

    radrange = radii.ptp()
    ring_width = 0.1
    ax.set_rlim(radrange * (1 - 1. / ring_width), radrange)
    ax.set_theta_offset(np.pi/2)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(['0', r'$\frac{\pi R}{2}$', r'$\pi R$', r'$\frac{3\pi R}{2}$'])
    ax.set_yticklabels([])
    ax.set_title('Density at {} (t={})'.format(title, str(round(t[time_index], 4))))
    fig.savefig(filename + '.png')

# time slice plots
density_time_slice(t_imag_start, 'imag start', 'density_imag_start')
density_time_slice(t_imag_end, 'imag end', 'density_imag_end')
density_time_slice(t_evo_end, 'evo end', 'density_evo_end')

# ring heatmaps
polar_heat(t_imag_start, 'imag start', 'ring_imag_start.png')
polar_heat(t_imag_end, 'imag end', 'ring_imag_end.png')
polar_heat(t_evo_end, 'evo end', 'ring_evo_end.png')

# plot functions
plot(x_3, psi_init, 'x', 'psi', 'initial wavefunction', 'psi')
plot(x_3, laser, 'x', 'V_g', 'gaussian laser potential', 'gaussian')