from typing import Optional

from ..nodes import *
from ..blocks import *

class SamplingGroupBlock(Block):
    """
    SamplingGroup block
    """
    def __init__(self, parent, basis, initial_sample):
        super().__init__(parent)
        self.basis = basis
        self.initial_sample = initial_sample

        self._head = SamplingGroupNode(self._parent)
        self._parent.add_child(self._head)
    
    def generate(self):
        # attributes
        if self.basis is not None:
            self._head.add_attribute('basis', self.basis)
        if self.initial_sample is not None:
            self._head.add_attribute('initial_sample', self.initial_sample)
        
        # equations
        self._head.text = self.equations_str()

        # comments
        if self.comment_str:
            self._head.comment = self.comment_str

        # components (moments)
        print(self.components)
        m = MomentsNode(self._head, self.components_str())
        self._head.add_child(m)

        # dependencies
        if self.dependencies:
            dep = DependenciesNode(self._head, self.dependencies_str())
            self._head.add_child(dep)