from .Node import Node

class FilterNode(Node) :
    def __init__(self, parent, is_cdata = True) :
        super().__init__('filter', parent, is_cdata=is_cdata)
    
    def validate(self) :
        # if self._attributes :
        #     return False
        if not self._text :
            return False
        if not self._is_cdata :
            return False
        # if not self._children :
        #     return False
        return True