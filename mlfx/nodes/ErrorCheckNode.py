from nodes import Node

class ErrorCheckNode(Node) :
    def __init__(self, parent) :
        super().__init__('error_check', parent)
    
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