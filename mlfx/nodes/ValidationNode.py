from .Node import Node

class ValidationNode(Node) :
    def __init__(self, parent, kind) :
        super().__init__('validation', parent, kind = kind)
    
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