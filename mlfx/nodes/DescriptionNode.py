from .Node import Node
from .SimulationNode import SimulationNode

class DescriptionNode(Node) :
    """
    Represents the description element in an XMDS2 file
    """
    def __init__(self, text: str, parent: SimulationNode) :
        """
        DescriptionNode constructor

        Args:
            text (str): The simulation description
            parent (SimulationNode): The simulation node
        """
        super().__init__('description', parent, text)
    
    def validate(self) :
        """
        Validates the node
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