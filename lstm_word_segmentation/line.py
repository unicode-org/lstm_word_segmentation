import numpy as np
from icu import BreakIterator, Locale, Char, UCharCategory
from bies import Bies
from collections import Counter


class Line:
    """
    A class that stores different verions of a line: unsegmented, ICU segmented, and manually segmented (if exists).
    Args:
        unsegmented: the unsegmented version of the line
        icu_segmented: the ICU segmented version of the line
        man_segmented: the manually segmented version of the line (doesn't always exist)
        icu_word_brkpoints: a list of word break points computed by ICU
        man_word_brkpoints: a list of word break points computed manually (doesn't always exist)
        char_brkpoints: a list of exgtended grapheme cluster breakpoints for the line
    """
    def __init__(self, input_line, input_type):
        """
        The __init__ function creates a new instance of the class based on the input line and its type.
        Args:
            input_line: the input line. It should be a clean line in one language.
            input_type: determines what is the type of the input line. It can be segmented, icu_segmented, or
            man_segmented.
        """
        if input_type == "unsegmented":
            self.unsegmented = input_line
            self._compute_char_brkpoints()
            self._compute_icu_segmented()
            self.icu_word_brkpoints = self._compute_word_brkpoints(self.icu_segmented)
            self.man_segmented = None
            self.man_word_brkpoints = None

        if input_type == "icu_segmented":
            self.icu_segmented = input_line
            self.icu_word_brkpoints = self._compute_word_brkpoints(self.icu_segmented)
            self.unsegmented = input_line.replace("|", "")
            self._compute_char_brkpoints()
            self.man_segmented = None
            self.man_word_brkpoints = None

        if input_type == "man_segmented":
            self.man_segmented = input_line
            self.man_word_brkpoints = self._compute_word_brkpoints(self.man_segmented)
            self.unsegmented = input_line.replace("|", "")
            self._compute_char_brkpoints()
            self._compute_icu_segmented()
            self.icu_word_brkpoints = self._compute_word_brkpoints(self.icu_segmented)

    def _compute_icu_segmented(self):
        """
        This function computes the ICU segmented version of the line.
        """
        words_break_iterator = BreakIterator.createWordInstance(Locale.getRoot())
        words_break_iterator.setText(self.unsegmented)
        self.icu_word_brkpoints = [0]
        for brkpoint in words_break_iterator:
            self.icu_word_brkpoints.append(brkpoint)
        self.icu_segmented = "|"
        for i in range(len(self.icu_word_brkpoints) - 1):
            self.icu_segmented += self.unsegmented[self.icu_word_brkpoints[i]: self.icu_word_brkpoints[i + 1]] + "|"

    def _compute_word_brkpoints(self, input_line):
        """
        Given a segmented line, this function computes a list of word breakpoints. Note that it treats all "|" as a
        separator between two successive words.
        Args:
            input_line: the input segmented line.
        """
        word_brkpoints = []
        found_bars = 0
        for i in range(len(input_line)):
            if input_line[i] == '|':
                word_brkpoints.append(i - found_bars)
                found_bars += 1
        return word_brkpoints

    def _compute_char_brkpoints(self):
        """
        This function uses ICU BreakIterator to identify and store extended grapheme clusters.
        """
        chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getRoot())
        chars_break_iterator.setText(self.unsegmented)
        self.char_brkpoints = [0]
        for brkpoint in chars_break_iterator:
            self.char_brkpoints.append(brkpoint)

    def get_bies(self, segmentation_type):
        """
        This function computes a matrix that represents BIES corresponding to the type.
        Args:
            segmentation_type: this can be "icu" or "man" which indicates which segmentation we want to be used for computing BIES
        """
        word_brkpoints = None
        if segmentation_type == "icu":
            word_brkpoints = self.icu_word_brkpoints
        if segmentation_type == "man":
            word_brkpoints = self.man_word_brkpoints
        if word_brkpoints is None:
            print("Warning! No segmentation exist for the given type.")
        bies_mat = np.zeros(shape=[len(self.char_brkpoints) - 1, 4])
        word_ind = 0
        for i in range(len(self.char_brkpoints) - 1):
            word_st = word_brkpoints[word_ind]
            word_fn = word_brkpoints[word_ind + 1]
            char_st = self.char_brkpoints[i]
            char_fn = self.char_brkpoints[i + 1]
            if char_st == word_st and char_fn != word_fn:
                bies_mat[i, 0] = 1
                continue
            if char_st != word_st and char_fn != word_fn:
                bies_mat[i, 1] = 1
                continue
            if char_st != word_st and char_fn == word_fn:
                bies_mat[i, 2] = 1
                word_ind += 1
                continue
            if char_st == word_st and char_fn == word_fn:
                bies_mat[i, 3] = 1
                word_ind += 1
                continue
        return Bies(input=bies_mat, input_type="mat")

    def get_grapheme_clusters(self):
        out = Counter()
        for i in range(len(self.char_brkpoints) - 1):
            new_graph_clust = self.unsegmented[self.char_brkpoints[i]: self.char_brkpoints[i + 1]]
            out[new_graph_clust] += 1
        return out

    def display(self):
        """
        This function displays different versions of the line (unsegmented, icu_segmented, man_segmented) and also word
        break points for different types of segmenters and a list of extended grapheme clusters in the line.
        """
        print("*******************************************************")
        print("unsegmented       : {}".format(self.unsegmented))
        print("icu_segmented     : {}".format(self.icu_segmented))
        print("man_segmented     : {}".format(self.man_segmented))
        print("icu_word_brkpoints: {}".format(self.icu_word_brkpoints))
        print("man_word_brkpoints: {}".format(self.man_word_brkpoints))
        print("char_brkpoints    : {}".format(self.char_brkpoints))
        print("*******************************************************")
