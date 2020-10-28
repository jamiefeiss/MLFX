from typing import Optional

from ..nodes import *
from ..blocks import *

class FilterBlock(Block):
    """
    Filter block

    Args:
        parent: The parent node (Simulation, Sequence or Filters)
        filter_name (str): The name of the filter

    Attributes:
        filter_name (str): The name of the filter
        name (str): The name of the block
        comment_str (str): The comment
        equations (List[str]): The list of equations
        dependencies (List[str])): The list of dependencies
    """
    def __init__(self, parent, filter_name: Optional[str] = None, in_integrate = False):
        super().__init__(parent)
        self.filter_name = filter_name
        self.filters_parent = parent
        self.in_integrate = in_integrate
        if not self.in_integrate:
            self._head = FilterNode(self.filters_parent, is_cdata=True)
            self.filters_parent.add_child(self._head)
    
    def add_eq(self, eq):
        self.equations.append(eq)
    
    def set_filters_parent(self, parent: FiltersNode):
        """
        Sets the immediate parent filters node

        Args:
            parent (filtersNode): The filters node
        """
        self.filters_parent = parent
    
    def generate(self):
        """Generates the filter block"""
        if self.in_integrate:
            self._head = FilterNode(self.filters_parent, is_cdata=True)
            self.filters_parent.add_child(self._head)

        # attributes
        if self.filter_name is not None:
            self._head.add_attribute('name', self.filter_name)

        # comments
        if self.comment_str:
            self._head.comment = self.comment_str

        # self._head.text = 

        # dependencies
        if self.dependencies:
            dep = DependenciesNode(self._head, self.dependencies_str())
            self._head.add_child(dep)
            dep.tail = self.equations_str()