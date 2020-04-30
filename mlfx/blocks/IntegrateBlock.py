from typing import Optional

from ..nodes import *
from ..blocks import *

class IntegrateBlock(Block):
    """
    Integrate block
    """
    def __init__(self, parent, algorithm, interval, steps = None, tolerance = None, samples = None):
        super().__init__(parent)
        self.algorithm = algorithm
        self.interval = interval
        self.steps = steps
        self.tolerance = tolerance
        self.samples = samples

        self.filters = []
        self.operators = []
    
    def add_filter(self)
        pass

    def add_operator(self)
        pass
    
    def generate(self):
        # integrate
        integrate = IntegrateNode(self.parent, self.algorithm, self,interval)

        # attributes
        if self.steps is not None:
            integrate.add_attribute('steps', self.steps)
        if self.tolerance is not None:
            integrate.add_attribute('tolerance', self.tolerance)
        
        self.parent.add_child(integrate)

        # comments
        if self.comment_str:
            integrate.comment = self.comment_str
        
        # filters

        # operators

        # dependencies

        # components (integration vectors)