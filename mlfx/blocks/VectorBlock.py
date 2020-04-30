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
    
    # init file
    
    def generate(self):
        # vector
        vec = VectorNode(self.parent, self.name)

        # attributes
        if self.type is not None:
            vec.add_attribute('type', self.type)
        if self.dimensions is not None:
            vec.add_attribute('dimensions', self.dimensions)
        if self.initial_basis is not None:
            vec.add_attribute('initial_basis', self.initial_basis)
        
        self.parent.add_child(vec)

        # comments
        if self.comment_str:
            vec.comment = self.comment_str

        # components
        c = ComponentsNode(vec, self.components_str())
        vec.add_child(c)

        # initialisation
        init = InitialisationNode(vec, self.equations_str(), True)
        vec.add_child(init)

        # dependencies
        if self.dependencies:
            dep = DependenciesNode(init, self.dependencies_str())
            init.add_child(dep)