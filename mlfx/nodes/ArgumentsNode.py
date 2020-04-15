from nodes import Node

class ArgumentsNode(Node) :
    def __init__(self, parent) :
        super().__init__('arguments', parent)
    
    def validate(self) :
        if self._attributes :
            return False
        if self._text and not self._is_cdata : # can only have CDATA
            return False
        if not self._children : # must have children
            return False
        return True