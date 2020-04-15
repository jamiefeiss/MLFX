from nodes import Node

class FFTWNode(Node) :
    def __init__(self, parent) :
        super().__init__('fftw', parent)
    
    def validate(self) :
        if not self._attributes :
            return False
        if self._text :
            return False
        if self._is_cdata :
            return False
        if self._children :
            return False
        return True