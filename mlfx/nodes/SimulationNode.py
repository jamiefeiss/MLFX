from lxml import etree

class SimulationNode :
    def __init__(self) :
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
    
    def validate(self) :
        '''Validates the node'''
        if not self._attributes :
            return False
        if not self._children :
            return False
        
        # check allowed/required (check child.name)

        return True
    
    def generate(self) :
        '''Generates the simulation node'''
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
    
    @property
    def attributes(self) :
        '''The attributes of the node'''
        return self._attributes
    
    def add_child(self, child) :
        '''Adds a child node'''
        self._children.append(child)