from .Node import Node
from .SimulationNode import SimulationNode

class AuthorNode(Node) :
    """
    Represents the author element in an XMDS2 file
    """
    def __init__(self, text: str, parent: SimulationNode) :
        """
        AuthorNode constructor

        Args:
            text (str): The author name
            parent (SimulationNode): The simulation node
        """
        super().__init__('author', parent, text)
    
    def validate(self) :
        """
        Validates the AuthorNode
        """
        if self._attributes :
            return False
        if self._is_cdata :
            return False
        if self._text is not str :
            return False
        if self._children :
            return False
        return True