from nodes import Node

class GlobalsNode(Node) :
    def __init__(self, parent) :
        super().__init__('globals', parent)
    
    def validate(self) :
        if self._attributes :
            return False
        if not self._text :
            return False
        if not self._is_cdata :
            return False
        if self._children :
            return False
        return True