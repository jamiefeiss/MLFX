from .Node import Node

class InitialisationNode(Node) :
    def __init__(self, parent, text, is_cdata = False) :
        super().__init__('initialisation', parent, text, is_cdata)
    
    def validate(self) :
        # if self._attributes :
        #     return False
        # if not self._text :
        #     return False
        # if self._is_cdata :
        #     return False
        # if self._children :
        #     return False
        return True