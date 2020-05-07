from typing import Optional

from ..nodes import *
from ..blocks import *

class VectorBlock(Block):
    """
    Vector block
    """
    def __init__(self, parent, type: Optional[str] = None, dimensions: Optional[str] = None, initial_basis: Optional[str] = None):
        super().__init__(parent)
        self.type = type
        self.dimensions = dimensions
        self.initial_basis = initial_basis

        self._head = VectorNode(self._parent, self.name)
        self._parent.add_child(self._head)
    
    # init file
    
    def generate(self):
        # attributes
        if self.type is not None:
            self._head.add_attribute('type', self.type)
        if self.dimensions is not None:
            self._head.add_attribute('dimensions', self.dimensions)
        if self.initial_basis is not None:
            self._head.add_attribute('initial_basis', self.initial_basis)

        # comments
        if self.comment_str:
            self._head.comment = self.comment_str

        # components
        c = ComponentsNode(self._head, self.components_str())
        self._head.add_child(c)

        # initialisation
        init = InitialisationNode(self._head, self.equations_str(), True)
        self._head.add_child(init)

        # dependencies
        if self.dependencies:
            dep = DependenciesNode(init, self.dependencies_str())
            init.add_child(dep)