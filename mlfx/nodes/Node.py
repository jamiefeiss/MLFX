from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, List, Dict, Optional
from lxml import etree

class Node(ABC):
    """
    Abstract class for an XMDS2 node
    """

    def __init__(self, tag: str, parent: Type[Node], text: Optional[str] = None, is_cdata: bool = False, **kwargs):
        """
        Node constructor

        Args:
            tag (str): The tag name of the node. Corresponds to the xml element tag
            parent (Node): The node's parent
            text (str): Text within the element. Can be CDATA if is_cdata is true
            is_cdata (bool): Flag to indicate whether the text within the element is CDATA or a string
            **kwargs: Attributes of the element
        """
        self._tag = tag
        self._parent = parent
        self._text = text
        self._is_cdata = is_cdata
        self._tail = '' # only for CDATA
        self._comment = ''
        self._attributes = {}
        self._children = []
        self.__allowed_children = []
        self.__allowed_attributes = {}
        self.__required_children = []
        self.__required_attributes = []

        for key, value in kwargs.items():
            self._attributes[key] = value

    @abstractmethod
    def validate(self):
        """Validates the node"""
        pass

    def generate(self):
        """
        Creates the node in XML
        
        First the node is validated, then the xml element is created and inserted into the tree.
        Then its attributes are inserted, then text.
        Lastly, each of its children are generated, in order.
        """
        self.validate()

        if self._comment:
            self._parent.element.append(etree.Comment(self._comment))

        self._element = etree.SubElement(self._parent.element, self._tag)

        for key, value in self._attributes.items():
            if value is not None:
                self._element.attrib[key] = value  

        if self._text:
            if self._is_cdata:
                self._element.text = etree.CDATA(self._text)
            else:
                self._element.text = self._text
        
        if self._tail:
            self._element.tail = etree.CDATA(self._tail)
        
        for child in self._children:
            child.generate()
    
    # @abstractmethod
    # @classmethod
    # def from_xml(self, xml):
    #     """Object creation from XML"""
    #     pass

    @property
    def element(self) -> etree._Element:
        """
        XML element object for the node
        
        Returns:
            The etree XML element
        """
        return self._element

    @property
    def tag(self) -> str:
        """
        The tag name of the node
        
        Returns:
            The tag name of the node (the xml element tag)
        """
        return self._tag
    
    @tag.setter
    def tag(self, tag: str):
        """
        Sets the tag name of the node
        
        Args:
            tag (str): The tag name of the node (the xml element tag)
        """
        self._tag = tag
    
    @property
    def parent(self) -> Type[Node]:
        """
        The parent of the node
        
        Returns:
            The parent of the node
        """
        return self._parent
    
    @parent.setter
    def parent(self, parent: Type[Node]):
        """
        Sets the parent of the node
        
        Args:
            parent (Node): The parent of the node
        """
        self._parent = parent
    
    @property
    def text(self) -> str:
        """
        The text of the node
        
        Returns:
            The text within the node
        """
        return self._text
    
    @text.setter
    def text(self, text: str):
        """
        Sets the text of the node
        
        Args:
            text (str): The text inside the node
        """
        self._text = text
    
    @property
    def is_cdata(self) -> bool:
        """
        Boolean flag for CDATA, true if text is CDATA
        
        Returns:
            The boolean flag of whether the node's text is CDATA
        """
        return self._is_cdata
    
    @is_cdata.setter
    def is_cdata(self, is_cdata: bool):
        """
        Sets flag for CDATA for the node
        
        Args:
            is_cdata (bool): Whether the text is CDATA or not
        """
        self._is_cdata = is_cdata
    
    @property
    def tail(self) -> str:
        """
        The tail of the node
        
        Returns:
            The tail after the node
        """
        return self._tail
    
    @tail.setter
    def tail(self, tail: str):
        """
        Sets the tail of the node
        
        Args:
            tail (str): The tail after the node
        """
        self._tail = tail
    
    @property
    def comment(self) -> str:
        """
        The comment of the node
        
        Returns:
            The comment of the node
        """
        return self._comment
    
    @comment.setter
    def comment(self, comment: str):
        """
        Sets the comment of the node
        
        Args:
            comment (str): The comment of the node
        """
        self._comment = comment

    @property
    def attributes(self) -> Dict:
        """
        The attributes of the node
        
        Returns:
            The node's attributes
        """
        return self._attributes
    
    @attributes.setter
    def attributes(self, attributes: Dict):
        """
        Sets the attributes of the node
        
        Args:
            attributes (Dict): The node's attributes
        """
        self._attributes = attributes
    
    def add_attribute(self, key, value):
        self._attributes.update({key: value})
    
    @property
    def children(self) -> List[Type[Node]]:
        """
        The children list of the node

        Returns:
            The node's children
        """
        return self._children

    def add_child(self, child: Type[Node]):
        """
        Adds a child node
        
        Args:
            child (Node): A child of the node
        """
        self._children.append(child)