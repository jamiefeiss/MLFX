from typing import Dict, Tuple, List, Optional, Union, Type

from lxml import etree
from plumbum import local
from plumbum.cmd import xmds2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from .nodes import *
from .blocks import *
from .variables import *
from NeuralNetwork import NeuralNetwork

class Project(object):
	"""Represents the XMDS2 simulation"""

	def __init__(self):
		self._node = SimulationNode()
		self._blocks = [] # [Block, ...]
		self._components = {} # {component: Block, ...}
		self._globals = [] # [Global, ....]
		self._parameters = [] # [Parameter, ...]
		self._no_params = 0
	
	def generate(self, filename: str):
		"""
		Generates the XMDS2 file
		
		Args:
			filename (str): The name of the output .xmds file
		"""
		# dependencies
		# scan over blocks for components
		for block in self._blocks:
			# don't register integrate components
			if isinstance(block, IntegrateBlock) or isinstance(block, SamplingGroupBlock):
				continue
			for component in block.components:
				self._components.update({component: block})

		# find other components used in equations
		for block in self._blocks:
			dependencies = []
			for eq in block.equations:
				for comp in self._components.keys():
					if comp in block.get_terms(block.get_rhs(eq)):
						dependencies.append(comp)
			if type(block) is IntegrateBlock:
				# find integration vectors
				for component in block.components:
					block.add_int_vec(self._components[component].name)
			if dependencies:
				for d in dependencies:
					if self._components[d].name in block.dependencies:
						continue
					if type(block) is IntegrateBlock:
						# dont have integration vectors as dependencies
						if self._components[d].name in block.int_vecs:
							continue
					block.add_dependency(self._components[d].name)
		
		# generate blocks
		for block in self._blocks:
			block.generate()

		# recursive call to validate & generate down the tree
		self._node.generate()

		# write to file
		self._tree = etree.ElementTree(self._node.element)
		self._tree.write(filename + '.xmds', pretty_print = True, xml_declaration = True, encoding = "UTF-8")
	
	def config(self, config: Dict):
		"""
		Sets the configuration for the simulation
		
		Args:
			config (Dict): Dictionary of the configuration
		"""
		if 'name' in config:
			self.new_name(config['name'])
			self.sim_name = config['name']
		if 'author' in config:
			self.new_author(config['author'])
		if 'description' in config:
			self.new_description(config['description'])
		
		# features
		features = FeaturesNode(self._node)
		self._node.add_child(features)
		# arguments
		if self._parameters:
			arguments = ArgumentsNode(features)
			features.add_child(arguments)
			for parameter in self._parameters:
				# a = ArgumentNode(arguments, argument[0], argument[1], argument[2])
				p = parameter.generate(arguments)
				arguments.add_child(p)
		if 'auto_vectorise' in config:
			if config['auto_vectorise'] == True:
				auto_vectorise = AutoVectoriseNode(features)
				features.add_child(auto_vectorise)
		if 'benchmark' in config:
			if config['benchmark'] == True:
				benchbark = BenchmarkNode(features)
				features.add_child(benchbark)
		if 'bing' in config:
			if config['bing'] == True:
				bing = BingNode(features)
				features.add_child(bing)
		# ignore cflags for now
		if 'chunked_output' in config:
			if config['chunked_output'] != False:
				chunked_output = ChunkedOutputNode(features, config['chunked_output'])
				features.add_child(chunked_output)
		if 'diagnostics' in config:
			if config['diagnostics'] == True:
				diagnostics = DiagnosticsNode(features)
				features.add_child(diagnostics)
		if 'error_check' in config:
			if config['error_check'] == True:
				error_check = ErrorCheckNode(features)
				features.add_child(error_check)
		if 'halt_non_finite' in config:
			if config['halt_non_finite'] == True:
				halt_non_finite = HaltNonFiniteNode(features)
				features.add_child(halt_non_finite)
		if 'fftw' in config:
			if config['fftw'] != False:
				if type(config['fftw']) is str:
					fftw = FFTWNode(features, config['fftw'])
				else: # tuple
					fftw = FFTWNode(features, config['fftw'][0], config['fftw'][1])
				features.add_child(fftw)
		if self._globals:
			g = GlobalsNode(features)
			g_str = '\n'
			for glob in self._globals:
				g_str += glob.generate()
				g_str += '\n'
			g.text = g_str
			features.add_child(g)
		if 'openmp' in config:
			if config['openmp'] != False:
				if config['openmp'] == True:
					openmp = OpenMPNode(features)
				else: # string
					openmp = OpenMPNode(features, config['openmp'])
				features.add_child(openmp)
		if 'precision' in config:
			if config['precision'] != False:
				precision = PrecisionNode(features, config['precision'])
				features.add_child(precision)
		if 'validation' in config:
			if config['validation'] != False:
				validation = ValidationNode(features, config['validation'])
				features.add_child(validation)
		
		# driver
		if 'driver' in config:
			if config['driver'] != False:
				if type(config['driver']) is str:
					driver = DriverNode(self._node, config['driver'])
				else: # tuple
					driver = DriverNode(self._node, config['driver'][0], config['driver'][1])
				self._node.add_child(driver)

		# geometry
		geometry = GeometryNode(self._node)
		self._node.add_child(geometry)
		if 'prop_dim' in config:
			propagation_dimension = PropagationDimensionNode(geometry, config['prop_dim'])
			geometry.add_child(propagation_dimension)
		if 'trans_dim' in config:
			transverse_dimensions = TransverseDimensionsNode(geometry)
			geometry.add_child(transverse_dimensions)
			# only consider lattice and domain for now
			for dim in config['trans_dim']:
				d = DimensionNode(transverse_dimensions, dim['name'], dim['lattice'], dim['domain'])
				transverse_dimensions.add_child(d)
		
		# neural network setup
		if 'ml_settings' in config:
			self.network = NeuralNetwork(parameters=self._parameters, settings=config['ml_settings'])
			self.network.construct_network()
			self.network.generate_training_input()

	def parse(self, xml):
		"""Parses an XMDS2 file"""
		# self._node.from_xml(xml)
		pass
	
	def compile(self, filename: str):
		"""
		Compiles the XMDS2 file
		
		Args:
			filename (str): The file name
		"""
		chain = xmds2[filename + '.xmds']
		chain()

	def run(self, param_vals: List[Tuple]):
		"""
		Runs the XMDS2 file with parameter values
		
		Args:
			sim_name (str): The name of the executable
			param_vals (List[Tuple]): A list of tuples containing parameter names & values
		"""
		# param_vals tuple (name, value)
		chain = local['./' + self.sim_name]
		for param in param_vals:
			chain('--' + param[0] + '=' + str(param[1]))
	
	def objective_function(self, x, cost_fn):
		# run sim
		param_vals = []
		for i in range(len(x)):
			param = [param for param in self._parameters if param.index == i]
			param_vals.append((param.name, x[i]))
		self.run(param_vals)

		# open h5 file
		f = h5py.File(self.sim_name + '.h5', 'r')
		dataset = f['1']
		output = dataset[self.cost_name]
		return cost_fn(output)
		
	def optimise(self, cost_fn): # requires user-defined cost function for post-processing output variable
		self.compile(self.filename)
		self.network.generate_training_output(self.objective_function, cost_fn)
		self.network.train()
		self.network.find_optimal_params()

	def optimise2(self, filename: str, sim_name: str, fig_name: str):
		"""
		Optimises over parameter set
		
		Args:
			filename (str): The file name
			sim_name (str): The name of the executable
			fig_name (str): The name the main output figure to be plotted
		"""
		# chain1 = xmds2[filename + '.xmds']
		# chain1()
		self.compile(filename)

		# k_min = 1.0
		# k_max = 2.0
		k_min = self._parameters[0].min
		k_max = self._parameters[0].max
		no_steps = 50
		step_size = (k_max - k_min) / no_steps
		o_max = 0
		k = k_min
		k_opt = k

		# loop
		while k <= k_max:
			self.run(sim_name, [(self._parameters[0].name, k)])
			# chain2 = local['./' + sim_name]
			# chain2('--k=' + str(k))

			f = h5py.File(sim_name + '.h5', 'r')
			dset3 = f['5']
			overlap = float(dset3['overlap1'][...])
			f.close()

			if overlap > o_max:
				o_max = overlap
				k_opt = k
			k += step_size
		
		self._parameters[0].set_optimal(k_opt)
		
		# optimal k
		print('k_opt=' + str(round(self._parameters[0].optimal, 4)))

		self.run(sim_name, [(self._parameters[0].name, self._parameters[0].optimal)])
		# chain2 = local['./' + sim_name]
		# chain2('--k=' + str(k_opt))

		self.plot(sim_name, fig_name)

	def plot(self, sim_name: str, fig_name: str):
		"""
		Creates plots
		
		Args:
			sim_name (str): The name of the executable
			fig_name (str): The name the main output figure to be plotted
		"""
		f = h5py.File(sim_name + '.h5', 'r')

		dset1 = f['1'] # psi
		dset2 = f['2'] # V
		dset3 = f['3'] # lambda
		dset4 = f['4'] # psi2
		dset5 = f['5'] # overlap

		d1 = dset1['density']
		x1 = dset1['x']

		p2 = dset2['p'] # V[t, x]
		x2 = dset2['x']

		l3 = dset3['l']
		t3 = dset3['t']

		d4 = dset4['density2']
		x4 = dset4['x']

		overlap = float(dset5['overlap1'][...])
		print('overlap=' + str(round(overlap, 4)))

		# final state density plot
		fig, ax = plt.subplots()  # Create a figure containing a single axes.
		ax.plot(x1[...], d1[...], label='final state')  # Plot some data on the axes.
		ax.plot(x4[...], d4[...], label='desired state')  # Plot some data on the axes.
		ax.set_xlabel('x')
		ax.set_ylabel('Density')
		ax.set_title('Final state density, ' + self._parameters[0].name + ' = ' + str(round(self._parameters[0].optimal, 4)) + ', overlap = ' + str(round(overlap, 4)))
		ax.legend()
		fig.savefig(fig_name + '.png')

		# timing function plot
		fig2, ax2 = plt.subplots()  # Create a figure containing a single axes.
		ax2.plot(t3[...], l3[...])  # Plot some data on the axes.
		ax2.set_xlabel('t')
		ax2.set_ylabel('lambda')
		ax2.set_title('Timing function, ' + self._parameters[0].name + ' = ' + str(round(self._parameters[0].optimal, 4)))
		fig2.savefig('lambda.png')

		# timing function plot
		fig3, ax3 = plt.subplots()  # Create a figure containing a single axes.
		ax3.plot(x2[...], p2[-1, ...])  # Plot some data on the axes.
		ax3.set_xlabel('x')
		ax3.set_ylabel('V')
		ax3.set_title('Potential, t=T')
		fig3.savefig('potential.png')
		
		f.close()
	
	def plot2(self, sim_name):
		# plots for shockwave density & fourier transform
		f = h5py.File(sim_name + '.h5', 'r')

		dset1 = f['1'] # x
		dset2 = f['2'] # k

		d1 = dset1['density']
		x1 = dset1['x']

		d2 = dset2['k_density']
		x2 = dset2['kx']

		# density plot
		fig1, ax1 = plt.subplots()
		ax1.plot(x1[...], d1[...])
		ax1.set_xlabel('x')
		ax1.set_ylabel('Density')
		ax1.set_title('State density')
		ax1.legend()
		fig1.savefig('density.png')

		# fourier density plot
		fig2, ax2 = plt.subplots()
		ax2.plot(x2[...], d2[...])
		ax2.set_xlabel('kx')
		ax2.set_ylabel('Density')
		ax2.set_title('Fourier transform of density')
		ax2.legend()
		fig2.savefig('k_density.png')

	### nodes

	def new_name(self, text: str):
		"""
		Adds a name
		
		Args:
			text (str): The name of the simulation
		"""
		name = NameNode(text, self._node)
		self._node.add_child(name)

	def new_author(self, text: str):
		"""
		Adds an author
		
		Args:
			text (str): The author's name
		"""
		author = AuthorNode(text, self._node)
		self._node.add_child(author)
	
	def new_description(self, text: str):
		"""
		Adds a description
		
		Args:
			text (str): The description of the simulation
		"""
		desc = DescriptionNode(text, self._node)
		self._node.add_child(desc)
	
	def new_vector(self, name: str, component: str, type: str, dimensions: str, initialisation: str):
		"""
		Adds a vector node

		Args:
			name (str): The vector name
			component (str): The component variable name
			type (str): The vector type
			dimensions (str): The dimensions of the vector
			initialisation (str): The initialisation equation
		"""
		# only consider type & dimensions, no dependencies
		vector = VectorNode(self._node, name, type, dimensions)
		# vector.comment = 'comment'
		self._node.add_child(vector)
		comp = ComponentsNode(vector, component)
		vector.add_child(comp)
		init = InitialisationNode(vector, initialisation, True)
		vector.add_child(init)
	
	def new_comp_vector(self, name: str, component: str, type: str, dimensions: str, evaluation: str):
		"""
		Adds a computed vector node

		Args:
			name (str): The computed vector name
			component (str): The component variable name
			type (str): The computed vector type
			dimensions (str): The dimensions of the computed vector
			evaluation (str): The evaluation equation
		"""
		# only consider type & dimensions, no dependencies
		comp_vector = ComputedVectorNode(self._node, name, type, dimensions)
		# vector.comment = 'comment'
		self._node.add_child(comp_vector)
		comp = ComponentsNode(comp_vector, component)
		comp_vector.add_child(comp)
		eval = EvaluationNode(comp_vector, evaluation, True)
		comp_vector.add_child(eval)
	
	# ignore filters

	### blocks
	
	def sequence(self) -> SequenceNode:
		"""Adds the main sequence node"""
		# ignore nested sequences for now
		self._sequence = SequenceNode(self._node)
		self._node.add_child(self._sequence)
		return self._sequence
	
	def output(self) -> OutputNode:
		"""Adds the output node"""
		self._output = OutputNode(self._node)
		self._node.add_child(self._output)
		return self._output
	
	def integrate(self, algorithm: str, interval: str, steps: Optional[str] = None, tolerance: Optional[str] = None, samples: Optional[str] = None) -> IntegrateBlock:
		"""
		Adds an integrate block

		Args:
			algorithm (str): The integration algorithm to be used
			interval (str): The time interval for the integration block
			steps (str): The number of steps in the integration
			tolerance (str): The integration tolerance for adaptive algorithms
			samples (str): The number of samples for each sampling group
		"""
		i = IntegrateBlock(self._sequence, algorithm, interval, steps, tolerance, samples)
		self._add_block(i)
		return i

	def vec(self, type: Optional[str] = None, dimensions: Optional[str] = None, initial_basis: Optional[str] = None) -> VectorBlock:
		"""
		Adds a vector block

		Args:
			type (str): The vector type
			dimensions (str): The vector dimensions
			initial_basis (str): The initial basis
		"""
		v = VectorBlock(self._node, type, dimensions, initial_basis)
		self._add_block(v)
		return v
	
	def comp_vec(self, type: Optional[str] = None, dimensions: Optional[str] = None, initial_basis: Optional[str] = None) -> ComputedVectorBlock:
		"""
		Adds a computed vector block

		Args:
			type (str): The vector type
			dimensions (str): The vector dimensions
			initial_basis (str): The initial basis
		"""
		cv = ComputedVectorBlock(self._node, type, dimensions, initial_basis)
		self._add_block(cv)
		return cv
	
	def operator(self, parent: IntegrateNode, kind: str, type: Optional[str] = None, constant: Optional[str] = None) -> OperatorBlock:
		"""
		Adds an operator block to an integration block

		Args:
			parent (IntegrateNode): The integrate node
			kind (str): The operator kind [ip/ex]
			type (str): The operator type
			constant (str): Whether the operator is constant [yes/no]
		"""
		o = OperatorBlock(parent, kind, type, constant)
		return o
	
	def sampling_group(self, basis: Optional[str] = None, initial_sample: Optional[str] = None) -> SamplingGroupBlock:
		"""
		Adds a sampling group block to the output node

		Args:
			basis (str): The dimension basis of the sampling group
			initial_sample (str): Whether the output is sampled before integration [yes/no]
		"""
		sg = SamplingGroupBlock(self._output, basis, initial_sample)
		self._add_block(sg)
		return sg
	
	def _add_block(self, block: Type[Block]):
		"""
		Adds a block to the simulation block list

		Args:
			block (Block): The block to be added
		"""
		self._blocks.append(block)
	
	# def add_global(self, glob):
	# 	self._globals.append(glob)

	def add_global(self, type: str, name: str, value: Union[int, float]):
		"""
		Adds a global to the simulation global list

		Args:
			type (str): The global type
			name (str): The global name
			value (int, float): The global value
		"""
		glob = Global(type, name, value)
		self._globals.append(glob)
	
	# def add_argument(self, name, type, default_value):
	# 	arg_tup = (name, type, default_value)
	# 	self._parameters.append(arg_tup)

	def parameter(self, type: str, name: str, default_value: Union[int, float], min: Union[int, float], max: Union[int, float]):
		"""
		Adds a parameter to the simulation parameter list

		Args:
			type (str): The parameter type
			name (str): The parameter name
			value (int, float): The parameter default value
			min (int, float): The parameter's minimum value
			max (int, float): The parameter's maximum value
		"""
		param = Parameter(type, name, default_value, min, max, index=self._no_params)
		self._parameters.append(param)
		self._no_params += 1
	
	def cost_variable(self, cost_variable: str):
		self.cost_name = cost_variable