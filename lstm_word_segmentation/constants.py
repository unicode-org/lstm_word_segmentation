import numpy as np
import os
from icu import BreakIterator, Locale, Char, UCharCategory

THAI_GRAPH_CLUST_RATIO = np.load(os.getcwd() + '/Data/Thai_graph_clust_ratio.npy', allow_pickle=True).item()

# BURMESE_GRAPH_CLUST_RATIO = np.load(os.getcwd() + '/Data/Burmese_graph_clust_ratio.npy', allow_pickle=True).item()


THAI_CHAR_TYPE_TO_BUCKET = {UCharCategory.UPPERCASE_LETTER: 1, UCharCategory.LOWERCASE_LETTER: 1,
                           UCharCategory.TITLECASE_LETTER: 1, UCharCategory.MODIFIER_LETTER: 1,
                           UCharCategory.OTHER_LETTER: 1, UCharCategory.NON_SPACING_MARK: 2,
                           UCharCategory.ENCLOSING_MARK: 2, UCharCategory.COMBINING_SPACING_MARK: 2,
                           UCharCategory.DECIMAL_DIGIT_NUMBER: 3, UCharCategory.LETTER_NUMBER: 3,
                           UCharCategory.OTHER_NUMBER: 3, UCharCategory.SPACE_SEPARATOR: 4,
                           UCharCategory.LINE_SEPARATOR: 4,
                           UCharCategory.PARAGRAPH_SEPARATOR: 4, UCharCategory.DASH_PUNCTUATION: 5,
                           UCharCategory.START_PUNCTUATION: 5, UCharCategory.END_PUNCTUATION: 5,
                           UCharCategory.CONNECTOR_PUNCTUATION: 5, UCharCategory.OTHER_PUNCTUATION: 5,
                           UCharCategory.INITIAL_PUNCTUATION: 5, UCharCategory.FINAL_PUNCTUATION: 5,
                           UCharCategory.MATH_SYMBOL: 6, UCharCategory.CURRENCY_SYMBOL: 6,
                           UCharCategory.MODIFIER_SYMBOL: 6,
                           UCharCategory.OTHER_SYMBOL: 6, UCharCategory.CONTROL_CHAR: 7, UCharCategory.FORMAT_CHAR: 7,
                           UCharCategory.PRIVATE_USE_CHAR: 7, UCharCategory.SURROGATE: 7, 0: 7}



