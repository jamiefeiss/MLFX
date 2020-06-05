from typing import Optional, List

from ..nodes import *
from ..blocks import *

class IntegrateBlock(Block):
    """
    Integrate block

    Args:
        parent (SequenceNode): The parent node
        algorithm (str): The integration algorithm to be used
        interval (str): The time interval for the integration block
        steps (str): The number of steps in the integration
        tolerance (str): The integration tolerance for adaptive algorithms
        samples (str): The number of samples for each sampling group

    Attributes:
        algorithm (str): The integration algorithm to be used
        interval (str): The time interval for the integration block
        steps (str): The number of steps in the integration
        tolerance (str): The integration tolerance for adaptive algorithms
        samples (str): The number of samples for each sampling group
        operators (List[OperatorBlock]): List of operator blocks
        int_vecs (List[str]): List of integration vectors
        name (str): The name of the block
        comment_str (str): The comment
        equations (List[str]): The list of equations
        components (List[str])): The list of components
        dependencies (List[str])): The list of dependencies
    """
    def __init__(self, parent: SequenceNode, algorithm: str, interval: str, steps: Optional[str] = None, tolerance: Optional[str] = None, samples: Optional[str] = None):
        super().__init__(parent)
        self.algorithm = algorithm
        self.interval = interval
        self.steps = steps
        self.tolerance = tolerance
        self.samples = samples

        # self.filters = [] # list of blocks
        self.operators = []
        self.int_vecs = []

        self._head = IntegrateNode(self._parent, self.algorithm, self.interval)
        self._parent.add_child(self._head)
    
    # def add_filter(self, filter):
    #     self.filters.append(filter)

    def add_operator(self, operator):
        """
        Adds an operator block to the operator list
        
        Args:
            operator (OperatorBlock): The operator block
        """
        self.operators.append(operator)
    
    def add_int_vec(self, int_vec: str):
        """
        Adds an integration vector to the int vec list
        
        Args:
            int_vec (str): The integration vector string
        """
        self.int_vecs.append(int_vec)
    
    def int_vecs_str(self) -> str:
        """
        Concatenates the integration vectors into a single string

        Returns:
            str: The concatenated integration vectors string
        """
        int_vec_str = ''
        for int_vec in self.int_vecs:
            int_vec_str += int_vec + ' '
        int_vec_str = int_vec_str.strip()

        return int_vec_str
    
    def get_int_vec_lhs(self, eq: str) -> str:
        """
        Extracts the integration vector from the left hand side of an equation
        
        Args:
            eq (str): The equation string
        
        Returns:
            str: The integration vector string
        """
        v = self.get_lhs(eq)
        v_str = v.split('_d')[0].split('d')[1]
        return v_str
    
    # override
    def add_eq(self, eq: str):
        """
        Adds an equation to the equation list
        
        Args:
            eq (str): The equation string
        """
        self.equations.append(eq)
        self.add_component(self.get_int_vec_lhs(eq))
    
    def generate(self):
        """Generates the integrate block"""
        # attributes
        if self.steps is not None:
            self._head.add_attribute('steps', self.steps)
        if self.tolerance is not None:
            self._head.add_attribute('tolerance', self.tolerance)

        # comments
        if self.comment_str:
            self._head.comment = self.comment_str
        
        # samples
        if self.samples:
            sam = SamplesNode(self._head, self.samples)
            self._head.add_child(sam)
        
        # # filters
        # if self.filters:
        #     fils = FiltersNode(self._head)
        #     self._head.add_child(fils)
        #     for filter in self.filters:
        #         print('filter')
        #         # filter.generate()

        # operators
        ops = OperatorsNode(self._head)
        self._head.add_child(ops)
        i = 0
        for operator in self.operators:
            operator.set_ops_parent(ops)
            # cdata after last operator
            if i + 1 == len(self.operators):
                operator.tail = self.equations_str()
            operator.generate()
            i += 1
        
        # integration vectors
        int_vecs = IntegrationVectorsNode(ops, self.int_vecs_str())
        ops.add_child(int_vecs)

        # dependencies
        if self.dependencies:
            dep = DependenciesNode(ops, self.dependencies_str())
            ops.add_child(dep)