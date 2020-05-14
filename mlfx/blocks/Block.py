from abc import ABC, abstractmethod
from typing import Type, Optional, List
import re

from varname import varname

from ..nodes import *

class Block(ABC):
    """
    Abstract Block class
    """

    def __init__(self, parent):
        self.name = varname(caller = 3)
        self._parent = parent
        self.comment_str = ''
        self.equations = []
        self.components = []
        self.dependencies = []

        # self._head
    
    def comment(self, comment):
        self.comment_str = comment
    
    def add_eq(self, eq):
        self.equations.append(eq)
        self.add_component(self.get_lhs(eq))
    
    def get_lhs(self, eq):
        return eq.split('=')[0].strip()
    
    def get_rhs(self, eq):
        return eq.split('=')[1].strip()
    
    def get_terms(self, eq):
        term_list = list(filter(None, re.split('[.,;\+\-\/\*()\[\]]', eq)))
        return [term.strip() for term in term_list]
    
    def components_str(self):
        c_str = ''
        for comp in self.components:
            c_str += comp + ' '
        c_str = c_str.strip()

        return c_str
    
    def equations_str(self):
        eq_str = '\n'
        for eq in self.equations:
            eq_str += eq
            eq_str += '\n'

        return eq_str
    
    def add_component(self, component):
        self.components.append(component)
    
    def add_dependency(self, dependency):
        self.dependencies.append(dependency)
    
    def dependencies_str(self):
        d_str = ''
        for dep in self.dependencies:
            d_str += dep + ' '
        d_str = d_str.strip()

        return d_str
    
    @abstractmethod
    def generate(self):
        pass