from nodes import Node

class AuthorNode(Node) :
    def __init__(self, text, parent) :
        super().__init__('author', parent, text)
    
    def validate(self) :
        if self._attributes :
            return False
        if self._is_cdata :
            return False
        if self._text is not str :
            return False
        if self._children :
            return False
        return True