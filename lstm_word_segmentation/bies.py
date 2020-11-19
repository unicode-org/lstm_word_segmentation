import numpy as np

class Bies:
    def __init__(self, input, input_type):
        if input_type == "mat":
            self.mat = input   # n * 4
            self.compute_str_from_mat()
        if input_type == "str":
            self.str = input
            self.mat = None

    def compute_str_from_mat(self):
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
        This function normalizes the input bies string to generate a bies string that makes sense. For example the output
        won't have substring such as "biiis", "biese" or "siie"
        Args:
            bies_str: The input bies string
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
