from typing import Optional

from .Node import Node

class FFTWNode(Node) :
    def __init__(self, parent, plan: str, threads: Optional[str] = None) :
        super().__init__('fftw', parent, plan = plan, threads = threads)
    
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