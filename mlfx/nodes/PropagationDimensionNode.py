from .Node import Node

class PropagationDimensionNode(Node) :
    def __init__(self, parent, text: str) :
        super().__init__('propagation_dimension', parent, text)
    
    def validate(self) :
        if self._attributes :
            return False
        if not self._text :
            return False
        if self._is_cdata :
            return False
        if self._children :
            return False
        return True