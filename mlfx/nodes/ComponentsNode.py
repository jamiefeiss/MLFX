from .Node import Node

class ComponentsNode(Node) :
    def __init__(self, parent, text) :
        super().__init__('components', parent, text)
    
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