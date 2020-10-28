from abc import ABC, abstractmethod
from typing import Type, Optional, List
import re

from varname import varname

from ..nodes import *

class Block(ABC):
    """
    Abstract Block class

    Args:
        parent (Node): The parent node

    Attributes:
        name (str): The name of the block
        comment_str (str): The comment
        equations (List[str]): The list of equations
        components (List[str])): The list of components
        dependencies (List[str])): The list of dependencies
    """

    def __init__(self, parent: Type[Node]):
        self.name = varname(caller = 3)
        self._parent = parent
        self.comment_str = ''
        self.equations = []
        self.components = []
        self.dependencies = []

        # self._head
    
    def comment(self, comment: str):
        """
        Adds a comment

        Args:
            comment (str): The comment string
        """
        self.comment_str = comment
    
    def add_eq(self, eq: str):
        """
        Adds an equation to the equation list
        
        Args:
            eq (str): The equation string
        """
        self.equations.append(eq)
        self.add_component(self.get_lhs(eq))
    
    def get_lhs(self, eq: str) -> str:
        """
        Extracts the left hand side of an equation string
        
        Args:
            eq (str): The equation string
        
        Returns:
            str: The left hand side of the equation
        """
        return eq.split('=')[0].strip()
    
    def get_rhs(self, eq: str) -> str:
        """
        Extracts the right hand side of an equation string
        
        Args:
            eq (str): The equation string
        
        Returns:
            str: The right hand side of the equation
        """
        return eq.split('=')[1].strip()
    
    def get_terms(self, eq: str) -> List[str]:
        """
        Extracts the terms of an equation
        
        Args:
            eq (str): The equation string
        
        Returns:
            List[str]: The list of term strings
        """
        term_list = list(filter(None, re.split('[.,;=\+\-\/\*()\[\]]', eq)))
        return [term.strip() for term in term_list]
    
    def components_str(self) -> str:
        """
        Concatenates the components into a single string

        Returns:
            str: The concatenated component string
        """
        c_str = ''
        for comp in self.components:
            c_str += comp + ' '
        c_str = c_str.strip()

        return c_str
    
    def equations_str(self) -> str:
        """
        Concatenates the equations into a single string

        Returns:
            str: The concatenated equation string
        """
        eq_str = '\n'
        for eq in self.equations:
            eq_str += eq
            eq_str += '\n'

        return eq_str
    
    def add_component(self, component: str):
        """
        Adds a component to the list of components

        Args:
            component (str): The component string
        """
        self.components.append(component)
    
    def add_dependency(self, dependency: str):
        """
        Adds a dependency to the list of dependencies

        Args:
            dependency (str): The dependency string
        """
        if dependency not in self.dependencies and not dependency == self.name:
            self.dependencies.append(dependency)
    
    def dependencies_str(self) -> str:
        """
        Concatenates the dependencies into a single string

        Returns:
            str: The concatenated dependencies string
        """
        d_str = ''
        for dep in self.dependencies:
            d_str += dep + ' '
        d_str = d_str.strip()

        return d_str
    
    @abstractmethod
    def generate(self):
        pass