from typing import Optional

from ..nodes import *
from ..blocks import *

class OperatorBlock(Block):
    """
    Operator block
    """
    def __init__(self, parent, kind: str, type: Optional[str] = None, constant: Optional[str] = None):
        super().__init__(parent)
        self.kind = kind
        self.type = type
        self.constant = constant
    
    def generate(self):
        # operator
        op = OperatorNode(self.parent, self.equations_str(), self.kind)

        # attributes
        if self.type is not None:
            op.add_attribute('type', self.type)
        if self.constant is not None:
            op.add_attribute('constant', self.constant)
        
        self.parent.add_child(op)

        # comments
        if self.comment_str:
            op.comment = self.comment_str

        # operator names
        op_names = OperatorNamesNode(op, self.components_str())
        op.add_child(op_names)

        # dependencies
        if self.dependencies:
            dep = DependenciesNode(op, self.dependencies_str())
            op.add_child(dep)