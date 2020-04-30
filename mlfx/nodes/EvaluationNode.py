from .Node import Node

class EvaluationNode(Node) :
    def __init__(self, parent, text, is_cdata = False) :
        super().__init__('evaluation', parent, text, is_cdata)
    
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