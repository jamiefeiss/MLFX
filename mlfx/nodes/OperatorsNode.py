from .Node import Node

class OperatorsNode(Node) :
    def __init__(self, parent, text) :
        super().__init__('operators', parent, text, is_cdata=True)
    
    def validate(self) :
        if self._attributes :
            return False
        if not self._text :
            return False
        if not self._is_cdata :
            return False
        if not self._children :
            return False
        return True