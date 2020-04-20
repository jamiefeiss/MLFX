from lxml import etree
from .nodes import *

class Project :
    '''Represents the XMDS2 simulation'''

    def __init__(self) :
        self._node = SimulationNode()
    
    def generate(self, filename) :
        '''Generates the XMDS2 file'''
        self._node.generate()

        self._tree = etree.ElementTree(self._node.element)
        self._tree.write(filename + '.xmds', pretty_print = True, xml_declaration = True, encoding = "UTF-8")
    
    def config(self, config) :
        '''Sets the configuration for the simulation'''
        pass
    
    def parse(self, xml) :
        '''Parses an XMDS2 file'''
        # self._node.from_xml(xml)
        pass

    def run(self) :
        '''Runs the XMDS2 file'''
        pass

    ### meta functions for generating nodes

    def new_author(self, author) :
        '''Adds an author'''
        author = AuthorNode('Jamie Feiss', self._node)
        self._node.add_child(author)