import numpy as np
from . import constants
from icu import Char


class CodePoint:
    """
    A class to store a code point. It supports the following versions of a code point:
        1) code_point_id: an integer that is computed with respect to a dictionary of all accepted code points
    """
    def __init__(self, codepoint, codepoint_dic):
        """
        The __init__ function creates a new instance of the class.
        Args:
            codepoint: the input code point
            codepoint_dic: the dictionary that stores all accepted codepoints in a language
        """
        self.codepoint = codepoint
        self.num_codepoints = len(codepoint_dic)+1
        self.codepoint_id = codepoint_dic.get(self.codepoint, self.num_codepoints - 1)

    def display(self):
        """
        A function that displays different versions of the current code point
        """
        print("Code point: \n{}".format(self.codepoint))
        print("Code point id: \n{}".format(self.codepoint_id))