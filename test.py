from mlfx import Project

p = Project()

config = {
    'name': 'name',
    'author': 'author',
    'description': 'desc',
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

p.config(config)

# p.new_author('Jamie Feiss')

# globals

init = 'psi = (1.0 / pow(M_PI, 0.25)) * exp(-pow(x, 2) / 2.0); // ground state of HO'

p.new_vector('wavefunction', 'psi', 'complex', 'x', init)

eval = 'lambda = t / T;'

p.new_comp_vector('timing_function', 'lambda', 'real', '', eval)

seq = p.sequence()

p.generate('filename')

