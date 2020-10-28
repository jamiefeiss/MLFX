import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

dset2 = f['2'] # fourier
k_density_re = dset2['k_density_re'][...]
k_density_im = dset2['k_density_im'][...]
kx = dset2['kx'][...]
tk = dset2['t'][...] - T_imag
k_density = k_density_re + 1j * k_density_im

def find_zero_gradient(f_mag, t, k_index):
    f_2k = np.abs(f_mag[..., k_index])
    max_indices = []
    f_grad = np.gradient(f_2k)
    for i in range(1, f_grad.shape[0] - 1):
        if (abs(f_grad[i - 1]) > abs(f_grad[i]) < abs(f_grad[i + 1])) and (f_grad[i - 1] > 0 > f_grad[i + 1]):
            max_indices.append(i)
    return max_indices

# samples
imag_samples = 0
evo_samples = 1000
t_imag_start = 0
t_imag_end = t_imag_start + imag_samples
t_evo_start = 0
t_evo_end = t_evo_start + evo_samples - 1

k_dom = 4
k_half = int(kx.shape[0]/2)
k_index = k_half + k_dom
max_indices = find_zero_gradient(k_density, tk, k_index)

t_max = []
f_max = []
f_plats = []
for index in max_indices:
    t_max.append(tk[index])
    f_max.append(np.abs(k_density[index, k_index]))
    f_plats.append(np.angle(k_density[index, k_index]))

start_index = 0
t_plat = []
plat_vals = []
for i in range(start_index, len(t_max)):
    t_plat.append(t_max[i])
    if f_plats[i] < 0:
        plat_vals.append(f_plats[i] + math.pi)
    else:
        plat_vals.append(f_plats[i])

# manual linear regression
def lin_reg(x, y):
    n = np.size(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    SS_xx = sum((x - x_mean)**2)
    SS_yy = sum((y - y_mean)**2)
    SS_xy = 0
    for i in range(n):
        SS_xy += (x[i] - x_mean) * (y[i] - y_mean)

    m = SS_xy / SS_xx # slope
    c = y_mean - m * x_mean # intercept

    def y_pred(x_i): # predicting using fit
        return m * x_i + c
    
    SS_E = 0
    for i in range(n):
        SS_E += (y[i] - y_pred(x[i]))**2

    # slope uncertainty
    m_uncert = math.sqrt(SS_E / ((n - 2) * (SS_xx)))
    return m, m_uncert

# linear regression
x = np.array(t_plat).reshape((-1, 1))
y = np.array(plat_vals)

# remove last element from fit
# x = x[:-1]
# y = y[:-1]

model = LinearRegression().fit(x, y)
y_pred = model.predict(x)

# gradient threshold
def phase_adjust(offset = 0):
    phase = np.angle(k_density[..., k_index]) + offset
    grad = np.gradient(phase)
    t_phase = []
    adjusted = []
    for i in range(len(phase)):
        if abs(grad[i]) < 0.05: # gradient threshold
            if 0.01 < phase[i] < 3.1:
                adjusted.append(phase[i])
                t_phase.append(tk[i])
    return t_phase, adjusted

# plot phase with linear fit
fig, ax = plt.subplots()
fig.set_size_inches(6.3, 3.5)
plt.subplots_adjust(bottom=0.2)
# ax.plot(phase_adjust()[0], phase_adjust()[1], '.', label='phase')
# ax.plot(phase_adjust(math.pi)[0], phase_adjust(math.pi)[1], '.', label='phase + ' + r'$\pi$')

plt.tick_params(labelsize=14)

ax.plot(tk, np.angle(k_density[..., k_index]), 'b-', label='Phase')
ax.plot(tk, np.angle(k_density[..., k_index]) + math.pi, 'b-')
ax.plot(t_plat, plat_vals, 'ro', label='Peaks')
ax.plot(x, y_pred, 'k-', label='Linear fit')
ax.set_xlabel('t', fontsize=18)
ax.set_ylabel('arg(' + r'$F_{2k}$' + ')', fontsize=18)
ax.set_ylim([0, math.pi])
ax.set_title('Linear fit of phase plateaus', fontsize=18)
ax.set_yticks([0, math.pi/2, math.pi])
ax.set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$'])
ax.legend(loc='lower right', fontsize=14)
fig.savefig('phase_fit_resize.png')

round_factor = 6

m, s_m = lin_reg(x, y)

print('y = {}x + {}, R^2={}'.format(round(model.coef_[0], round_factor), round(model.intercept_, round_factor), round(model.score(x, y), round_factor)))
print('sklearn m/k={}'.format(model.coef_[0]/(k_dom/R)))
print('m={}, {}'.format(m[0], s_m))
print('m/k={}, {}'.format(m[0]/(k_dom/R), s_m/(k_dom/R)))