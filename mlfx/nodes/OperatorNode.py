from .Node import Node

class OperatorNode(Node) :
    def __init__(self, parent, text, kind, type = None, constant = None) :
        super().__init__('operator', parent, text, True, kind = kind, type = type, constant = constant)
    
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