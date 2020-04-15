from nodes import Node

class OpenMPNode(Node) :
    def __init__(self, parent) :
        super().__init__('openmp', parent)
    
    def validate(self) :
        if self._text :
            return False
        if self._is_cdata :
            return False
        if self._children :
            return False
        return True