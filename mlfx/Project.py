from typing import Dict

from lxml import etree
from plumbum import local
from plumbum.cmd import xmds2, python3
import h5py
import matplotlib.pyplot as plt
import numpy as np

from .nodes import *
from .blocks import *

class Project(object):
	"""Represents the XMDS2 simulation"""

	def __init__(self):
		self._node = SimulationNode()
		self._blocks = []
		self._components = {} # {component: block}, ...
		self._globals = []
		self._arguments = []
	
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
			# if type(block) is IntegrateBlock or SamplingGroupBlock:
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
		"""Sets the configuration for the simulation"""
		if 'name' in config:
			self.new_name(config['name'])
		if 'author' in config:
			self.new_author(config['author'])
		if 'description' in config:
			self.new_description(config['description'])
		
		# features
		features = FeaturesNode(self._node)
		self._node.add_child(features)
		# arguments
		if self._arguments:
			arguments = ArgumentsNode(features)
			features.add_child(arguments)
			for argument in self._arguments:
				a = ArgumentNode(arguments, argument[0], argument[1], argument[2])
				arguments.add_child(a)
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
				g_str += glob
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

	def parse(self, xml):
		"""Parses an XMDS2 file"""
		# self._node.from_xml(xml)
		pass

	def run(self, filename, sim_name, fig_name):
		"""Runs the XMDS2 file"""
		# compile
		chain1 = xmds2[filename + '.xmds']
		chain1()

		k_min = 1.0
		k_max = 2.0
		no_steps = 50
		step_size = (k_max - k_min) / no_steps
		o_max = 0
		k = k_min
		k_opt = k

		# loop
		while k <= k_max:
			chain2 = local['./' + sim_name]
			chain2('--k=' + str(k))

			f = h5py.File(sim_name + '.h5', 'r')
			dset3 = f['5']
			overlap = float(dset3['overlap1'][...])
			f.close()

			if overlap > o_max:
				o_max = overlap
				k_opt = k
			k += step_size

		# k_opt = 1
		
		# optimal k
		print('k_opt=' + str(round(k_opt, 4)))

		chain2 = local['./' + sim_name]
		chain2('--k=' + str(k_opt))

		f = h5py.File(sim_name + '.h5', 'r')

		dset1 = f['1']
		dset2 = f['2']
		dset3 = f['3']
		dset4 = f['4']
		dset5 = f['5']

		d1 = dset1['density']
		x1 = dset1['x']

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
		ax.set_title('Final state density, k = ' + str(round(k_opt, 4)) + ', overlap = ' + str(round(overlap, 4)))
		ax.legend()
		fig.savefig(fig_name + '.png')

		# timing function plot
		fig2, ax2 = plt.subplots()  # Create a figure containing a single axes.
		ax2.plot(t3[...], l3[...])  # Plot some data on the axes.
		ax2.set_xlabel('t')
		ax2.set_ylabel('lambda')
		ax2.set_title('Timing function, k = ' + str(round(k_opt, 4)))
		fig2.savefig('lambda.png')
		
		f.close()

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
		# only consider type & dimensions, no dependencies
		vector = VectorNode(self._node, name, type, dimensions)
		# vector.comment = 'comment'
		self._node.add_child(vector)
		comp = ComponentsNode(vector, component)
		vector.add_child(comp)
		init = InitialisationNode(vector, initialisation, True)
		vector.add_child(init)
	
	def new_comp_vector(self, name: str, component: str, type: str, dimensions: str, evaluation: str):
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
	
	def sequence(self):
		# ignore nested sequences for now
		self._sequence = SequenceNode(self._node)
		self._node.add_child(self._sequence)
		return self._sequence
	
	def output(self):
		self._output = OutputNode(self._node)
		self._node.add_child(self._output)
		return self._output
	
	def integrate(self, algorithm, interval, steps = None, tolerance = None, samples = None):
		i = IntegrateBlock(self._sequence, algorithm, interval, steps, tolerance, samples)
		self._add_block(i)
		return i

	def vec(self, type = None, dimensions = None, initial_basis = None):
		v = VectorBlock(self._node, type, dimensions, initial_basis)
		self._add_block(v)
		return v
	
	def comp_vec(self, type = None, dimensions = None, initial_basis = None):
		cv = ComputedVectorBlock(self._node, type, dimensions, initial_basis)
		self._add_block(cv)
		return cv
	
	def operator(self, parent, kind, type = None, constant = None):
		o = OperatorBlock(parent, kind, type, constant)
		return o
	
	def sampling_group(self, basis = None, initial_sample = None):
		sg = SamplingGroupBlock(self._output, basis, initial_sample)
		self._add_block(sg)
		return sg
	
	def _add_block(self, block):
		self._blocks.append(block)
	
	def add_global(self, glob):
		self._globals.append(glob)
	
	def add_argument(self, name, type, default_value):
		arg_tup = (name, type, default_value)
		self._arguments.append(arg_tup)