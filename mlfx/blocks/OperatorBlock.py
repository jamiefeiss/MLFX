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
        self.tail = ''

    def set_ops_parent(self, parent):
        self.ops_parent = parent
    
    def generate(self):
        self._head = OperatorNode(self.ops_parent, self.kind)
        self.ops_parent.add_child(self._head)

        # equations
        self._head.text = self.equations_str()

        if self.tail:
            self._head.tail = self.tail

        # attributes
        if self.type is not None:
            self._head.add_attribute('type', self.type)
        if self.constant is not None:
            self._head.add_attribute('constant', self.constant)

        # comments
        if self.comment_str:
            self._head.comment = self.comment_str

        # operator names
        op_names = OperatorNamesNode(self._head, self.components_str())
        self._head.add_child(op_names)

        # dependencies
        if self.dependencies:
            dep = DependenciesNode(self._head, self.dependencies_str())
            self._head.add_child(dep)