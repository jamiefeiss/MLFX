from .Node import Node

class DependenciesNode(Node) :
    def __init__(self, parent, text) :
        super().__init__('dependencies', parent, text)
    
    def validate(self) :
        # if self._attributes :
        #     return False
        if not self._text :
            return False
        if self._is_cdata :
            return False
        if self._children :
            return False
        return True