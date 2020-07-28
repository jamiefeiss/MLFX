from typing import Dict, Union

from .Variable import Variable
from ..nodes import *

class Parameter(Variable):
    """
    Parameter variable
    
    Args:
        type (str): The variable type
        name (str): The variable name
        value: (int, float): The variable value
        min: (int, float): The variable's minimum value
        max: (int, float): The variable's maximum value

    Attributes:
        type (str): The variable type
        name (str): The variable name
        value: (int, float): The variable value
        min: (int, float): The variable's minimum value
        max: (int, float): The variable's maximum value
        comment (str): The variable comment
        optimal: (int, float): The variable's optimal value
    """
    def __init__(self, type: str, name: str, value: Union[int, float], min: Union[int, float], max: Union[int, float], index: int):
        super().__init__(type, name, value)
        self.min = min
        self.max = max
        self.index = index
    
    def set_optimal(self, optimal: Union[int, float]):
        """
        Sets the optimal value
        
        Args:
            optimal (int, float): The optimal value of the parameter
        """
        self.optimal = optimal
    
    def validate(self):
        """Validates the parameter"""
        pass
    
    def generate(self, parent: ArgumentsNode) -> ArgumentNode:
        """
        Generates the parameter
        
        Args:
            parent (ArgumentsNode): The parent ArgumentsNode
        
        Returns:
            ArgumentNode: The node to be added to arguments
        """
        # ignore comments for now
        parameter_node = ArgumentNode(parent, self.name, self.type, str(self.value))

        return parameter_node