from typing import Type, Dict, List

from lxml import etree

from .Node import Node

class SimulationNode :
    """
    Represents the whole XMDS2 simulation.

    The root element of the xml tree.
    """
    def __init__(self) :
        """
        SimulationNode constructor
        """
        self._name = 'simulation'
        self._attributes = {
            'xmds-version': '2'
        }
        self._children = []

        self.__allowed_children = [
            'name',\
            'author',\
            'description',\
            'features',\
            'driver',\
            'geometry',\
            'vector',\
            'noise_vector',\
            'computed_vector',\
            'filter',\
            'sequence',\
            'output'
        ]
        self.__allowed_attributes = {
            'xmds-version': '2'
        }
        self.__required_children = [
            'geometry',\
            'vector',\
            'sequence',\
            'output'
        ]
        self.__required_attributes = [
            'xmds-version'
        ]
    
    def validate(self) -> bool :
        """Validates the node"""
        if not self._attributes :
            return False
        if not self._children :
            return False
        
        # check allowed/required (check child.name)

        return True
    
    def generate(self) :
        """Generates the simulation node"""
        if self.validate() :
            print('valid')
        else :
            print('invalid')

        self._element = etree.Element(self._name)

        for key, value in self._attributes.items() :
            self._element.attrib[key] = value
        
        for child in self._children :
            child.generate()
    
    @classmethod
    def from_xml(self, xml) :
        """Object creation from XML"""
        pass

    @property
    def element(self) -> etree._Element :
        """
        XML element object for the node
        
        Returns:
            The etree XML element
        """
        return self._element
    
    @property
    def name(self) -> str :
        """
        The name of the node
        
        Returns:
            The name of the node (the xml element tag)
        """"
        return self._name
    
    @property
    def attributes(self) -> Dict :
        """
        The attributes of the node
        
        Returns:
            The node's attributes
        """
        return self._attributes
    
    @property
    def children(self) -> List[Type[Node]]:
        """
        The children list of the node

        Returns:
            The node's children
        """
        return self._children

    def add_child(self, child: Type[Node]) :
        """
        Adds a child node
        
        Args:
            child (Node): A child of the node
        """
        self._children.append(child)