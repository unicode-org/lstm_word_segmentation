import numpy as np


class Bies:
    """
    A class that stores a bies sequence in it.
    """
    def __init__(self, input_bies, input_type):
        """
        The __init__ function creates a new instance of the class.
        Args:
            input_bies: the input to initialize the instance
            input_type: determines what is the type of the input. It can be "mat" or "str"
        """
        if input_type == "mat":
            self.mat = input_bies
            self.compute_str_from_mat()
        elif input_type == "str":
            self.str = input_bies
            self.mat = None
        else:
            print("Warning: this input_type is not known for BIES")

    def compute_str_from_mat(self):
        """
        This function uses the matrix format of the bies sequence to generate a string version.
        """
        self.str = ""
        for i in range(self.mat.shape[0]):
            max_ind = np.argmax(self.mat[i, :])
            if max_ind == 0:
                self.str += "b"
            elif max_ind == 1:
                self.str += "i"
            elif max_ind == 2:
                self.str += "e"
            elif max_ind == 3:
                self.str += "s"

    def normalize_bies(self):
        """
        This function normalizes the BIES sequence to generate a string that makes sense. For example the output
        won't have substring such as "biiis", "biese" or "siie". Note that it doesn't change the self.mat
        """
        if len(self.str) == 0:
            return 's'
        out_bies = ""
        start_of_word = True
        skip = False
        for i in range(len(self.str)):
            if skip:
                skip = False
                continue
            if start_of_word:
                if i == len(self.str) - 1 or self.str[i] == 's':
                    out_bies += 's'
                    start_of_word = True
                    continue
                elif self.str[i] == 'b':
                    out_bies += self.str[i]
                    start_of_word = False
                    continue
                elif self.str[i] in ['i', 'e']:
                    if self.str[i + 1] in ['i', 'e']:
                        out_bies += 'b'
                        start_of_word = False
                        continue
                    else:
                        out_bies += 's'
                        start_of_word = True
                        continue
            if not start_of_word:
                if self.str[i] == 'e':
                    out_bies += self.str[i]
                    start_of_word = True
                    continue
                elif self.str[i] in ['b', 'i', 's']:
                    if i == len(self.str) - 1 or self.str[i + 1] in ['b', 's']:
                        out_bies += 'e'
                        start_of_word = True
                        continue
                    else:
                        out_bies += 'i'
                        start_of_word = False
                        continue
        self.str = out_bies
