from .Node import Node

class VectorNode(Node) :
    def __init__(self, parent, name: str) :
        super().__init__('vector', parent, name = name)
    
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