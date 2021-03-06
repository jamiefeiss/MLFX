from .Node import Node

class DimensionNode(Node) :
    def __init__(self, parent, name: str, lattice: str, domain: str) :
        super().__init__('dimension', parent, name = name, lattice = lattice, domain = domain)
    
    def validate(self) :
        if not self._attributes :
            return False
        if self._text :
            return False
        if self._is_cdata :
            return False
        if self._children :
            return False
        return True