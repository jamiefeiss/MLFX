from .Node import Node

class SamplingGroupNode(Node) :
    def __init__(self, parent) :
        super().__init__('sampling_group', parent, is_cdata=True)
    
    def validate(self) :
        if not self._attributes :
            return False
        if not self._text :
            return False
        if not self._is_cdata :
            return False
        if not self._children :
            return False
        return True