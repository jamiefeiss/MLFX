from .Node import Node

class OutputNode(Node) :
    def __init__(self, parent) :
        super().__init__('output', parent)
    
    def validate(self) :
        # if not self._attributes :
        #     return False
        if self._text :
            return False
        if self._is_cdata :
            return False
        if not self._children :
            return False
        return True