from .Node import Node

class DiagnosticsNode(Node) :
    def __init__(self, parent) :
        super().__init__('diagnostics', parent)
    
    def validate(self) :
        if self._attributes :
            return False
        if self._text :
            return False
        if self._is_cdata :
            return False
        if self._children :
            return False
        return True