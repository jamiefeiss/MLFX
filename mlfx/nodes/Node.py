from abc import ABC, abstractmethod
from lxml import etree

class Node(ABC) :
    '''
    Abstract class for an XMDS2 node

    Args:
        name (str):
        parent (etree.SubElement):
        text (str):
        is_cdata (bool):
        **kwargs:
    
    Attributes:
        name (str):
        parent (etree.SubElement):
        text (str):
        is_cdata (bool):
        attributes (dict):
        element (etree.SubElement):
    '''

    def __init__(self, name, parent, text = None, is_cdata = False, **kwargs) :
        self._name = name
        self._parent = parent
        self._text = text
        self._is_cdata = is_cdata
        self._attributes = {}
        self._children = []
        # comment?
        # allowed children
        # self.__allowed_children = []
        # allowed attributes (& values)
        # self.__allowed_attributes = {}
        # required children
        # self.__required_children = []
        # required attributes
        # self.__required_attributes = []

        for key, value in kwargs.items() :
            self._attributes[key] = value

    @abstractmethod
    def validate(self) :
        '''Validates the node'''
        pass

    def generate(self) :
        '''Creates the node in XML'''
        self.validate()

        self._element = etree.SubElement(self._parent.element(), self._name)

        for key, value in self._attributes.items() :
            self._element.attrib[key] = value

        if self._is_cdata :
            self._element.text = etree.CDATA(self._text)
        else :
            self._element.text = self._text
        
        # generate children
    
    @abstractmethod
    @classmethod
    def from_xml(self, xml) :
        '''Object creation from XML'''
        pass

    @property
    def element(self) :
        '''XML element object for the node'''
        return self._element

    @property
    def name(self) :
        '''The name of the node'''
        return self._name
    
    @name.setter
    def name(self, name) :
        '''Sets the name of the node'''
        self._name = name
    
    @property
    def parent(self) :
        '''The parent of the node'''
        return self._parent
    
    @parent.setter
    def parent(self, parent) :
        '''Sets the parent of the node'''
        self._parent = parent
    
    @property
    def text(self) :
        '''The text of the node'''
        return self._text
    
    @text.setter
    def text(self, text) :
        '''Sets the text of the node'''
        self._text = text
    
    @property
    def is_cdata(self) :
        '''Boolean flag for CDATA, true if text is CDATA'''
        return self._is_cdata
    
    @is_cdata.setter
    def is_cdata(self, is_cdata) :
        '''Sets flag for CDATA for the node'''
        self._is_cdata = is_cdata
    
    @property
    def attributes(self) :
        '''The attributes of the node'''
        return self._attributes
    
    @attributes.setter
    def attributes(self, attributes) :
        '''Sets the attributes of the node'''
        self._attributes = attributes