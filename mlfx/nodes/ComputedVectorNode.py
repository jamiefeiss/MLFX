from .Node import Node

class ComputedVectorNode(Node) :
    def __init__(self, parent, name: str, type: str, dimensions: str) :
        super().__init__('computed_vector', parent, name = name, type = type, dimensions = dimensions)
    
    def validate(self) :
        if not self._attributes :
            return False
        if self._text :
            return False
        if self._is_cdata :
            return False
        if not self._children :
            return False
        return True