from lxml import etree
import nodes

class Project :
    '''Represents the XMDS2 simulation'''

    def __init__(self) :
        self._node = nodes.SimulationNode()
    
    def generate(self, filename) :
        '''Generates the XMDS2 file'''
        # validate all nodes

        self._node.generate()

        self._tree = etree.ElementTree(self._node.element())
        self._tree.write(filename + '.xmds', pretty_print = True, xml_declaration = True, encoding = "UTF-8")

        # generate children
    
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

    # newNode()