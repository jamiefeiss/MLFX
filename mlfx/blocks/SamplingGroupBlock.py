from ..nodes import *
from ..blocks import *

class SamplingGroupBlock(Block):
    """
    SamplingGroup block

    Args:
        parent (OutputNode): The parent node
        basis (str): The output dimension basis
        initial_sample (str): Whether the output is sampled initially

    Attributes:
        basis (str): The output dimension basis
        initial_sample (str): Whether the output is sampled initially
        name (str): The name of the block
        comment_str (str): The comment
        equations (List[str]): The list of equations
        components (List[str])): The list of components
        dependencies (List[str])): The list of dependencies
    """
    def __init__(self, parent: OutputNode, basis: str, initial_sample: str):
        super().__init__(parent)
        self.basis = basis
        self.initial_sample = initial_sample
        self.comp_vecs = []

        self._head = SamplingGroupNode(self._parent)
        self._parent.add_child(self._head)
    
    def add_comp_vec(self, comp_vec: Type[ComputedVectorBlock]):
        """
        Adds a ComputedVectorBlock
        
        Args:
            comp_vec (ComputedVectorBlock): The ComputedVectorBlock to be added
        """
        self.comp_vecs.append(comp_vec)
    
    def generate(self):
        """Generates the sampling group block"""
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
        
        if self.comp_vecs:
            for comp_vec in self.comp_vecs:
                self._head.add_child(comp_vec)
                comp_vec.generate()

        # components (moments)
        m = MomentsNode(self._head, self.components_str())
        self._head.add_child(m)

        # dependencies
        # if self.dependencies:
        dep = DependenciesNode(self._head, self.dependencies_str())
        self._head.add_child(dep)