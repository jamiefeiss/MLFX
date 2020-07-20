from .Node import Node

class IntegrateNode(Node) :
    def __init__(self, parent, algorithm, interval) :
        super().__init__('integrate', parent, algorithm = algorithm, interval = interval)
    
    def validate(self) :
        if not self._attributes :
            return False
        if self._text :
            return False
        if self._is_cdata :
            return False
        if not self._children :
            return False
        return True