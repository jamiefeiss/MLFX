from typing import Dict, Union

from .Variable import Variable

class Global(Variable):
    """
    Global variable
    
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
        super().__init__(type, name, value)
    
    def validate(self):
        """Validates the global"""
        pass
    
    def generate(self) -> str:
        """
        Generates the global
        
        Returns:
            str: The string format of the global to be added to the globals node
        """
        global_str = 'const ' + self.type + ' ' + self.name + ' = ' + str(self.value) + ';'
        if self.comment:
            global_str += ' // ' + self.comment
        
        return global_str