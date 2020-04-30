from .Node import Node

class FeaturesNode(Node) :
    def __init__(self, parent) :
        super().__init__('features', parent)
    
    def validate(self) :
        if self._attributes :
            return False
        if self._text :
            return False
        if self._is_cdata :
            return False
        if not self._children : # must have children
            return False
        return True