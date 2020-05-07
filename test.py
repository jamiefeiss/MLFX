from mlfx import Project

p = Project()

config = {
    'name': 'bec_transport_generated',
    'author': 'Jamie Feiss',
    'description': 'Testing generating the BEC transport problem using the library',
    # 'arguments': [],
    'auto_vectorise': True,
    'benchmark': True,
    # 'bing': False,
    # 'cflags': False,
    # 'chunked_output': False,
    # 'diagnostics': True,
    # 'error_check': True,
    # 'halt_non_finite': True,
    'fftw': 'patient',
    # 'openmp': False,
    # 'precision': False,
    'validation': 'run-time',
    # 'driver': False,
    'prop_dim': 't',
    'trans_dim': [
        {
            'name': 'x',
            'lattice': '64',
            'domain': '(-10, 10)'
        }
    ]
}

p.add_global('const real T = 3.0; // End time')
p.add_global('const real x_0 = 3.0; // Final position')

p.config(config)

wavefunction = p.vec(type = 'complex', dimensions = 'x')
wavefunction.comment('Wavefunction')
init = 'psi = (1.0 / pow(M_PI, 0.25)) * exp(-pow(x, 2) / 2.0); // ground state of HO'
wavefunction.add_eq(init)

timing_function = p.comp_vec()
timing_function.type = 'real'
timing_function.dimensions = ''
timing_function.comment('Timing function')
timing_function.add_eq('lambda = t / T;')

potential = p.vec()
potential.type = 'real'
potential.dimensions = 'x'
potential.comment('Harmonic trap potential')
potential.add_eq('V = pow(x - lambda * x_0, 2) / 2.0;')

seq = p.sequence()

imag_time = p.integrate('RK4', '5.0e-3', '1000')
imag_time.add_eq('dpsi_dt = Ltt[psi] - (V + mod2(psi)) * psi;')
imag_time.comment('imaginary time')
op1 = p.operator(imag_time._head, 'ip', 'real', 'yes')
op1.add_eq('Ltt = -pow(kx, 2) / 2.0;')
imag_time.add_operator(op1)

gpe = p.integrate('ARK45', 'T', tolerance = '1e-8', samples = '100')
gpe.add_eq('dpsi_dt = Ltt[psi] - i * (V + mod2(psi)) * psi;')
gpe.comment('gpe')
op2 = p.operator(gpe._head, 'ip', 'imaginary', 'yes')
op2.add_eq('Ltt = -i * pow(kx, 2) / 2.0;')
gpe.add_operator(op2)

o = p.output()

s = p.sampling_group('x', 'no')
s.add_eq('psi_real = psi.Re();')
s.add_eq('psi_imag = psi.Im();')
s.add_eq('density = mod2(psi);')

p.generate('filename')

