from .Node import Node

class PrecisionNode(Node) :
    def __init__(self, parent, text: str) :
        super().__init__('precision', parent, text)
    
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