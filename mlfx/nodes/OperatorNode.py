from .Node import Node

class OperatorNode(Node) :
    def __init__(self, parent, kind, type = None, constant = None) :
        super().__init__('operator', parent, is_cdata = True, kind = kind, type = type, constant = constant)
    
    def validate(self) :
        if not self._attributes :
            return False
        if not self._text :
            return False
        if not self._is_cdata :
            return False
        if not self._children :
            return False
        return True