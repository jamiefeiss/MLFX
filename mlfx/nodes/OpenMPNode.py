from typing import Optional

from .Node import Node

class OpenMPNode(Node) :
    def __init__(self, parent, threads: Optional[str] = None) :
        super().__init__('openmp', parent, threads = threads)
    
    def validate(self) :
        if self._text :
            return False
        if self._is_cdata :
            return False
        if self._children :
            return False
        return True