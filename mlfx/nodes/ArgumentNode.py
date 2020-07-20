from .Node import Node

class ArgumentNode(Node) :
    def __init__(self, parent, name, type, default_value) :
        super().__init__('argument', parent, name = name, type = type, default_value = default_value)

        # self.__allowed_attributes = {
        #     'name': '',
        #     'type': [
        #         'real',
        #         'integer',
        #         'double',
        #         'int',
        #         'long'
        #     ],
        #     'default_value': ''
        # }

        # self.__required_attributes = [
        #     'name',
        #     'type',
        #     'default_value'
        # ]
    
    def validate(self) :
        if not self._attributes :
            return False
        if self._text :
            return False
        if self._is_cdata :
            return False
        if self._children :
            return False
        return True