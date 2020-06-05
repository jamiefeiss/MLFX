from abc import ABC, abstractmethod
from typing import Dict, Union

class Variable(ABC):
    """
    Abstract variable class

    Args:
        type (str): The variable type
        name (str): The variable name
        value: (int, float): The variable value

    Attributes:
        type (str): The variable type
        name (str): The variable name
        value: (int, float): The variable value
        comment (str): The variable comment
    """
    def __init__(self, type: str, name: str, value: Union[int, float]):
        self.type = type
        self.name = name
        self.value = value
        self.comment = ''
    
    # allowed types?

    def comment(self, comment: str):
        """
        Adds a comment to the variable
        
        Args:
            comment (str): The comment
        """
        self.comment = comment
    
    @abstractmethod
    def validate(self):
        pass
    
    @abstractmethod
    def generate(self):
        pass