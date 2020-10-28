import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from mlfx import Project

p = Project()

R = 10
p.add_global('real', 'R', R)
Omega = 0.05
p.add_global('real', 'Omega', Omega)
p.add_global('real', 'delta', Omega / 10)
L = 2 * math.pi * R
p.add_global('real', 'L', L)

N = 10
p.add_global('real', 'N', N)
p.add_global('real', 'g', 1.0)
p.add_global('real', 'A_psi', 0.9)
p.add_global('real', 'w_psi', 3.0)

p.add_global('real', 'A', 0.3)
p.add_global('real', 'w', 1.5)

p.add_global('real', 'T_imag', 1000)
p.add_global('real', 'T_evo', 160)

config = {
    'xmds_settings': {
        'name': 'shockwave_unopt',
        'author': 'Jamie Feiss',
        'description': 'Shockwave interferometer in 1D, unoptimised',
        'auto_vectorise': True,
        'benchmark': True,
        'fftw': 'patient',
        'validation': 'run-time',
        'prop_dim': 't',
        'trans_dim': [
            {
                'name': 'x',
                'lattice': '100',
                'domain': '(' + str(-L/4) + ',' + str(3*L/4) + ')'
            }
        ]
    }
}

p.config(config)

wavefunction = p.vec(type='complex', dimensions='x')
wavefunction.comment('Wavefunction')
wavefunction.add_eq('psi = A_psi * exp(-pow(x, 2.0) / (2 * pow(w_psi, 2.0)));')
wavefunction.add_eq('psi_plus = A_psi * exp(-pow(x, 2.0) / (2 * pow(w_psi, 2.0))); // Omega + delta')
wavefunction.add_eq('psi_minus = A_psi * exp(-pow(x, 2.0) / (2 * pow(w_psi, 2.0))); // Omega - delta')

normalisation = p.comp_vec(type = 'real', dimensions = '')
normalisation.add_eq('norm = mod2(psi); // calculate wavefunction normalisation')
normalisation.add_eq('norm_plus = mod2(psi_plus);')
normalisation.add_eq('norm_minus = mod2(psi_minus);')

gaussian = p.comp_vec(type = 'real', dimensions = 'x')
gaussian.add_eq('V_g = -A * exp(-pow(x, 2.0) / (2 * pow(w, 2.0)));')

seq = p.sequence()

filter1 = p.filter(seq)
filter1.add_eq('psi *= sqrt(N/norm);')
filter1.add_eq('psi_plus *= sqrt(N/norm_plus);')
filter1.add_eq('psi_minus *= sqrt(N/norm_minus);')
filter1.comment('Normalisation')

imag_time = p.integrate('ARK45', 'T_imag', steps='10000', tolerance='1e-10', samples='0 0 0 0 0')
op1 = p.operator(imag_time._head, 'ip', 'real', 'yes')
op1.add_eq('Ltt = -pow(kx, 2.0) / 2.0 + Omega * kx;')
op1.add_eq('Ltt_plus = -pow(kx, 2.0) / 2.0 + (Omega + delta) * kx;')
op1.add_eq('Ltt_minus = -pow(kx, 2.0) / 2.0 + (Omega - delta) * kx;')
imag_time.add_operator(op1)
imag_time.add_eq('dpsi_dt = Ltt[psi] - (V_g + g * mod2(psi)) * psi;')
imag_time.add_eq('dpsi_plus_dt = Ltt_plus[psi_plus] - (V_g + g * mod2(psi_plus)) * psi_plus;')
imag_time.add_eq('dpsi_minus_dt = Ltt_minus[psi_minus] - (V_g + g * mod2(psi_minus)) * psi_minus;')
imag_time.comment('Imaginary time')

filter2 = p.filter(imag_time._head, in_integrate=True)
filter2.add_eq('psi *= sqrt(N/norm);')
filter2.add_eq('psi_plus *= sqrt(N/norm_plus);')
filter2.add_eq('psi_minus *= sqrt(N/norm_minus);')
filter2.comment('Normalisation')
imag_time.add_filter(filter2)

gpe = p.integrate('ARK45', 'T_evo', tolerance='1e-8', samples='1000 1000 1000 0 0')
op2 = p.operator(gpe._head, 'ip', 'imaginary', 'yes')
op2.add_eq('Ltt = -i * pow(kx, 2.0) / 2.0 + i * Omega * kx;')
op2.add_eq('Ltt_plus = -i * pow(kx, 2.0) / 2.0 + i * (Omega + delta) * kx;')
op2.add_eq('Ltt_minus = -i * pow(kx, 2.0) / 2.0 + i * (Omega - delta) * kx;')
gpe.add_operator(op2)
gpe.add_eq('dpsi_dt = Ltt[psi] - i * (g * mod2(psi)) * psi;')
gpe.add_eq('dpsi_plus_dt = Ltt_plus[psi_plus] - i * (g * mod2(psi_plus)) * psi_plus;')
gpe.add_eq('dpsi_minus_dt = Ltt_minus[psi_minus] - i * (g * mod2(psi_minus)) * psi_minus;')
gpe.comment('GPE for shockwaves')

o = p.output()

s1 = p.sampling_group(basis='x', initial_sample='no')
s1.add_eq('psi_re = psi.Re();')
s1.add_eq('psi_im = psi.Im();')
s1.add_eq('density = mod2(psi);')
s1.comment('wavefunction & density')

s2 = p.sampling_group(basis='kx', initial_sample='no')

density_wavefunction = p.comp_vec(s2._head, type='complex', dimensions='x')
density_wavefunction.add_eq('density = mod2(psi);')
s2.add_comp_vec(density_wavefunction)

s2.add_eq('k_density_re = density.Re();')
s2.add_eq('k_density_im = density.Im();')
s2.comment('fourier density')

s3 = p.sampling_group(basis='x(0)', initial_sample='no')

fq_terms = p.comp_vec(s3._head, type='complex', dimensions='x')
fq_terms.add_eq('psi_diff = (psi_plus - psi_minus) / (2 * delta);')
fq_terms.add_eq('density_diff = (mod2(psi_plus) - mod2(psi_minus)) / (2 * delta);')
fq_terms.add_eq('term_1 = conj(psi_diff) * psi_diff;')
fq_terms.add_eq('term_2 = conj(psi) * psi_diff;')
fq_terms.add_eq('f_c = pow(density_diff, 2) / mod2(psi);')
s3.add_comp_vec(fq_terms)

s3.add_eq('F_Q_1 = term_1.Re(); // 1st term')
s3.add_eq('F_Q_2_re = term_2.Re(); // 2nd term')
s3.add_eq('F_Q_2_im = term_2.Im(); // 2nd term')
s3.add_eq('F_C = N * f_c.Re(); // classical fisher info')
s3.comment('integration for Fisher info')

s4 = p.sampling_group(basis='', initial_sample='yes')
s4.add_eq('radius = R; // ring radius')
s4.add_eq('omega = Omega; // rotation rate')
s4.add_eq('d_omega = delta; // small Omega change for differentiation')
s4.add_eq('length = L; // length of line')
s4.add_eq('no_atoms = N; // particle number')
s4.add_eq('phi = 2.0 * Omega * M_PI * pow(R, 2.0); // Sagnac phase-shift')
s4.add_eq('non_lin = g; // non-linearity constant')
s4.add_eq('amplitude_psi = A_psi; // gaussian beam amplitude')
s4.add_eq('width_psi = w_psi; // gaussian beam width')
s4.add_eq('amplitude = A; // gaussian beam amplitude')
s4.add_eq('width = w; // gaussian beam width')
s4.add_eq('t_imag = T_imag; // imaginary time')
s4.add_eq('t_evo = T_evo; // evolution time for shockwaves')
s4.comment('constants')

s5 = p.sampling_group(basis='x', initial_sample='yes')
s5.add_eq('psi_init = psi.Re();')
s5.add_eq('laser = V_g;')
s5.comment('functions')

p.generate('xmds_shockwave_unopt')
p.compile('xmds_shockwave_unopt')
p.run()