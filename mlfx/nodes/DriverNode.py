from typing import Optional

from .Node import Node

class DriverNode(Node) :
    def __init__(self, parent, name: str, paths: Optional[str] = None) :
        super().__init__('driver', parent, name = name, paths = paths)
    
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