import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# fourier transform
k_density_re = dset2['k_density_re'][...]
k_density_im = dset2['k_density_im'][...]
kx = dset2['kx'][...]
tk = dset2['t'][...]

k_density = k_density_re + 1j * k_density_im

# constants
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

# functions
psi_init = dset5['psi_init'][...]
laser = dset5['laser'][...]
x_3 = dset5['x'][...]

# number of samples per integration block
imag_samples = 0
evo_samples = 10000

# time indicies per integration block
t_imag_start = 0
t_imag_end = t_imag_start + imag_samples

# t_evo_start = t_imag_end + 1
t_evo_start = 0
t_evo_end = t_evo_start + evo_samples - 1


def plot(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([-1, 1])
    ax.set_title(title)
    fig.savefig(filename + '.png')

def find_maximums(f_mag, t, k_index):
    f_2k = np.abs(f_mag[..., k_index])
    max_indices = []
    for i in range(t.shape[0] - 1):
        if f_2k[i-1] < f_2k[i] > f_2k[i+1]:
            max_indices.append(i)
    return max_indices
      
def find_zero_gradient(f_mag, t, k_index):
    f_2k = np.abs(f_mag[..., k_index])
    max_indices = []
    f_grad = np.gradient(f_2k)
    for i in range(1, f_grad.shape[0] - 1):
        if (abs(f_grad[i - 1]) > abs(f_grad[i]) < abs(f_grad[i + 1])) and (f_grad[i - 1] > 0 > f_grad[i + 1]):
            max_indices.append(i)
    return max_indices

# plot fourier transform


# dominant k
k_dom = 4
k_half = int(kx.shape[0]/2)
k_index = k_half + k_dom

print('Omega={}'.format(Omega))
print('k={}'.format(k_dom))

def unwrap(func):
    # return np.unwrap(2 * func) / 2
    return np.unwrap(func)

phase = np.angle(k_density[..., k_index])
phase_unwrapped = unwrap(phase)

# phase_unwrapped = -np.unwrap(2*phase)/2 + math.pi

# max_indices = find_maximums(k_density, tk, k_index)
max_indices = find_zero_gradient(k_density, tk, k_index)

t_max = []
f_max = []
f_plats = []
for index in max_indices:
    t_max.append(tk[index])
    f_max.append(np.abs(k_density[index, k_index]))
    f_plats.append(phase_unwrapped[index])

# fourier density mag at dominant k slice
fig1, ax1 = plt.subplots()
ax1.plot(tk, np.abs(k_density[..., k_index]))
ax1.plot(t_max, f_max, 'r.')
ax1.set_xlabel('t')
ax1.set_ylabel('|F|')
ax1.set_title('Fourier density at 2k={}'.format(str(round(kx[k_index], 4))))
fig1.savefig('fourier_kslice.png')

round_factor = 6

# build arrays for linear regression
start_index = 0

t_plat = []
plat_vals = []
for i in range(start_index, len(t_max)):
    t_plat.append(t_max[i])
    plat_vals.append(f_plats[i])

# find delta t
delta_t = []
for i in range(len(t_plat) - 1):
    dt = t_plat[i + 1] - t_plat[i]
    delta_t.append(dt)

def list_avg(x):
    return sum(x) / len(x)

# print('delta_t = {}'.format([round(value, round_factor) for value in delta_t]))
print('delta_t average = {}'.format(round(list_avg(delta_t), round_factor)))

# find delta phi
delta_phi = []
for i in range(len(plat_vals) - 1):
    dphi = plat_vals[i + 1] - plat_vals[i]
    delta_phi.append(dphi)

# print('delta_phi = {}'.format([round(value, round_factor) for value in delta_phi]))
print('delta_phi average = {}'.format(round(list_avg(delta_phi), round_factor)))
# print('sagnac={}'.format(phi))

# linear regression
x = np.array(t_plat).reshape((-1, 1))
y = np.array(plat_vals)
model = LinearRegression().fit(x, y)
# print('y = {}x + {}, R^2={}'.format(round(model.coef_[0], round_factor), round(model.intercept_, round_factor), round(model.score(x, y), round_factor)))
print('slope = {}'.format(round(model.coef_[0], round_factor)))
y_pred = model.predict(x)

# fourier density phase at dominant k slice

def k_offset(k):
    unwrapped = unwrap(np.angle(k_density[..., k_half + k]))
    if not k % 2 == 0:
        unwrapped = unwrapped - math.pi
    if unwrapped[0] < -6:
        unwrapped = unwrapped + 2 * math.pi
    return unwrapped

k_plot = 4

fig2, ax2 = plt.subplots()
# ax2.plot(tk, phase, label='phase')
# ax2.plot(tk, phase + math.pi, label='phase + pi')
# ax2.plot(tk, k_offset(k_dom), label=k_dom)
# ax2.plot(tk, k_offset(k_plot), label=k_plot)
ax2.plot(tk, np.angle(k_density[..., k_half + k_plot]), label=k_plot)
ax2.plot(tk, np.angle(k_density[..., k_half + k_plot]) + math.pi, label=k_plot)
# ax2.plot(tk, k_offset(k_plot+1), label=k_plot+1) # odd k's offset by pi
# ax2.plot(tk, k_offset(k_plot+2), label=k_plot+2)
# ax2.plot(tk, k_offset(k_plot+3), label=k_plot+3)
# ax2.plot(tk, k_offset(k_plot+4), label=k_plot+4)
# ax2.plot(tk, k_offset(k_plot+5), label=k_plot+5)
# ax2.plot(tk, k_offset(k_plot+6), label=k_plot+6)
# ax2.plot(tk, k_offset(k_plot+7), label=k_plot+7)
# ax2.plot(t_plat, plat_vals, 'r.', label='peaks')
# ax2.plot(t_plat, [value if k_dom % 2 == 0 else value - math.pi for value in plat_vals], 'r.', label='peaks')
# ax2.plot(x, y_pred, 'k-', label='fit')
ax2.set_xlabel('t')
ax2.set_ylabel('F_arg')
# ax2.set_ylim([-math.pi, math.pi])
# ax2.set_ylim([-15, 0])
ax2.set_title('Fourier density phase at 2k={}, Omega={}'.format(str(round(kx[k_index], 4)), Omega))
ax2.legend()
fig2.savefig('fourier_phase_plateau.png')

# interval = list_avg(delta_phi) / (2 * math.pi**3 * R)
factor = 2 * math.pi * R**2
slope = model.coef_[0] / (2 * R**2 * k_dom/2)

# print('Omega: {}'.format(Omega))
# print('interval: {}'.format(round(interval, round_factor)))
# print('plats: {}'.format(plat_vals))
# print('plats/factor: {}'.format([round(value / factor, round_factor) for value in plat_vals]))
# print('slope/factor: {}'.format(round(slope, round_factor)))
# print(plat_vals[0])

# print([round(value, round_factor) if k_dom % 2 == 0 else round(value - math.pi, round_factor) for value in plat_vals])