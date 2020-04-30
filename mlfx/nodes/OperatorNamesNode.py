from .Node import Node

class OperatorNamesNode(Node) :
    def __init__(self, parent) :
        super().__init__('operator_names', parent)
    
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