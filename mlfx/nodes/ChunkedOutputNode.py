from .Node import Node

class ChunkedOutputNode(Node) :
    def __init__(self, parent, size: str) :
        super().__init__('chunked_output', parent, size = size)
    
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