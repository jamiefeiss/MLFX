import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
import gif

f = h5py.File('shockwave.h5', 'r')

dset1 = f['1']
dset2 = f['2']
dset3 = f['3']
dset4 = f['4']
dset5 = f['5']

# density
P = dset1['density'][...]
psi_re = dset1['psi_re']
psi_im = dset1['psi_im']
x = dset1['x'][...]
t = dset1['t'][...]

# constants
R = dset4['radius'][...]
Omega = dset4['omega'][...]
L = dset4['length'][...]
g = dset4['non_lin'][...]
A_psi = dset4['amplitude_psi'][...]
w_psi = dset4['width_psi'][...]
A = dset4['amplitude'][...]
w = dset4['width'][...]
T_imag = dset4['t_imag'][...]
T_evo = dset4['t_evo'][...]

# functions
psi_init = dset5['psi_init'][...]
laser = dset5['laser'][...]
x_3 = dset5['x'][...]

# number of samples per integration block
imag_samples = 0
evo_samples = 1000

# time indicies per integration block
t_imag_start = 0
t_imag_end = t_imag_start + imag_samples

# t_evo_start = t_imag_end + 1
t_evo_start = 0
t_evo_end = t_evo_start + evo_samples - 1

# x[x<0] += L

# gif animation
@gif.frame
def plot_frame(i):
    plt.ylim([0, 0.3])
    plt.xlabel(r'$\frac{x}{\pi}$')
    plt.ylabel(r'$|\psi|^2$')
    # if i < 1000:
    #     plt.title('Imaginary time (t={})'.format(round(t[i], 4)))
    #     plt.plot(x/(L/2), P[i], 'r-')
    #     # plt.plot(x/(L/2), psi_re[i], 'b-')
    #     # plt.plot(x/(L/2), psi_im[i], 'r-')
    #     # plt.legend(['Real', 'Imag'])
    # else:
    #     plt.title('Free evolution (t={})'.format(round(t[i], 4)))
    #     plt.plot(x/(L/2), P[i], 'b-')
    #     # plt.plot(x/(L/2), psi_re[i], 'c-')
    #     # plt.plot(x/(L/2), psi_im[i], 'm-')
    #     # plt.legend(['Real', 'Imag'])
    # plt.title('Free evolution (t={})'.format(round(t[i], 4)))
    plt.title('Free evolution (t_i={})'.format(i))
    plt.plot(x/(L/2), P[i], 'b-')
    # plt.plot(x/(L/2), psi_re[i], 'c-')
    # plt.plot(x/(L/2), psi_im[i], 'm-')
    # plt.legend(['Real', 'Imag'])

frames = []
no_steps = 100
step_size = int(round(len(t)/no_steps))
for i in range(0, len(t) - 1, step_size):
    frame = plot_frame(i)
    frames.append(frame)

gif.save(frames, 'test.gif', duration=100)