from typing import Dict

from lxml import etree
from .nodes import *
from .blocks import *

class Project(object):
	"""Represents the XMDS2 simulation"""

	def __init__(self):
		self._node = SimulationNode()
		self._blocks = []
		self._components = {} # {component: block}, ...
	
	def generate(self, filename: str):
		"""
		Generates the XMDS2 file
		
		Args:
			filename (str): The name of the output .xmds file
		"""
		# dependencies
		# scan over blocks for components
		for block in self._blocks:
			for component in block.components:
				self._components.update({component: block})

		# find other components used in equations
		for block in self._blocks:
			dependencies = []
			for eq in block.equations:
				for comp in self._components.keys():
					if comp in block.get_rhs(eq):
						dependencies.append(comp)
			if dependencies:
				for d in dependencies:
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
		# ignore arguments for now
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
		# ignore globals for now
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

	def run(self):
		"""Runs the XMDS2 file"""
		pass

	### meta functions for generating nodes

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
	
	def sequence(self):
		seq = SequenceNode(self._node)
		self._node.add_child(seq)
		return seq
	
	# def integrate(self, parent, )

	def vec(self, type = None, dimensions = None, initial_basis = None):
		v = VectorBlock(self._node, type, dimensions, initial_basis)
		self._add_block(v)
		return v
	
	def operator(self, kind, type = None, constant = None):
		o = OperatorBlock(self._node, kind, type, constant)
		self._add_block(o)
		return o
	
	def _add_block(self, block):
		self._blocks.append(block)