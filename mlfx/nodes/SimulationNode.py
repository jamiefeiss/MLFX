from lxml import etree

class SimulationNode :
    def __init__(self) :
        self._name = 'simulation'
        self._attributes = {
            'xmds-version': '2'
        }
        self._children = []
    
    def validate(self) :
        if not self._attributes :
            return False
        if not self._children :
            return False
        return True
    
    def generate(self) :
        '''Generates the XMDS2 file'''
        # validate all nodes
        if self.validate() :
            print('valid')
        else :
            print('invalid')

        self._element = etree.Element(self._name)

        for key, value in self._attributes.items() :
            self._element.attrib[key] = value
        
        # generate children
    
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