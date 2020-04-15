from .Node import Node

class EvaluationNode(Node) :
    def __init__(self, parent) :
        super().__init__('evaluation', parent)
    
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