from .Node import Node

class NameNode(Node) :
    def __init__(self, text, parent) :
        super().__init__('name', parent, text)
    
    def validate(self) :
        if self._attributes :
            return False
        if self._is_cdata :
            return False
        if self._text is not str :
            return False
        if self._children :
            return False
        return True