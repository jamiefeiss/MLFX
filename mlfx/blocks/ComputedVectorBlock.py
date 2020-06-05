from typing import Optional

from ..nodes import *
from ..blocks import *

class ComputedVectorBlock(Block):
    """
    ComputedVector block

    Args:
        parent (SimulationNode): The parent node
        type (str): The computed vector type
        dimensions (str): The computed vector dimensions
        initial_basis (str): The initial basis

    Attributes:
        type (str): The computed vector type
        dimensions (str): The computed vector dimensions
        initial_basis (str): The initial basis
        name (str): The name of the block
        comment_str (str): The comment
        equations (List[str]): The list of equations
        components (List[str])): The list of components
        dependencies (List[str])): The list of dependencies
    """
    def __init__(self, parent: SimulationNode, type: Optional[str] = None, dimensions: Optional[str] = None, initial_basis: Optional[str] = None):
        super().__init__(parent)
        self.type = type
        self.dimensions = dimensions
        self.initial_basis = initial_basis

        self._head = ComputedVectorNode(self._parent, self.name)
        self._parent.add_child(self._head)
    
    def generate(self):
        """Generates the computed vector block"""
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

        # initialisation (evaluation)
        init = EvaluationNode(self._head, self.equations_str(), True)
        self._head.add_child(init)

        # dependencies
        if self.dependencies:
            dep = DependenciesNode(init, self.dependencies_str())
            init.add_child(dep)