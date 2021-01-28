import numpy as np
from pathlib import Path
from icu import UCharCategory, UnicodeSet

# The dictionary that stores grapheme clusters and the frequency they appeared in BEST data set
path = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Public_Data/Thai_graph_clust_ratio.npy')
THAI_GRAPH_CLUST_RATIO = np.load(str(path), allow_pickle=True).item()

# The dictionary that stores only Thai-script grapheme clusters and the frequency they appeared in BEST data set
path = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Public_Data/Thai_exclusive_graph_clust_ratio.npy')
THAI_EXCLUSIVE_GRAPH_CLUST_RATIO = np.load(str(path), allow_pickle=True).item()

# The dictionary for all code points in Unicode Thai boxes
accepted_code_points = UnicodeSet("[[:Thai:]&[:LineBreak=SA:]]")
THAI_CODE_POINT_DICTIONARY = {accepted_code_points[i]: i for i in range(len(accepted_code_points))}

# The dictionary that stores grapheme clusters and the frequency they appeared in "my" data set
path = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Public_Data/Burmese_graph_clust_ratio.npy')
BURMESE_GRAPH_CLUST_RATIO = np.load(str(path), allow_pickle=True).item()

# The dictionary that stores only Burmese-script grapheme clusters and the frequency they appeared in "my" data set
path = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Public_Data/Burmese_exclusive_graph_clust_ratio.npy')
BURMESE_EXCLUSIVE_GRAPH_CLUST_RATIO = np.load(str(path), allow_pickle=True).item()

# The dictionary for all code points in Unicode Burmese boxes
accepted_code_points = UnicodeSet("[[:Mymr:]&[:LineBreak=SA:]]")
BURMESE_CODE_POINT_DICTIONARY = {accepted_code_points[i]: i for i in range(len(accepted_code_points))}

# The dictionary that stores grapheme clusters and the frequency they appeared in BEST and my data sets
path = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Public_Data/Thai_Burmese_graph_clust_ratio.npy')
THAI_BURMESE_GRAPH_CLUST_RATIO = np.load(str(path), allow_pickle=True).item()

# A dictionary that determines how different types of code points are grouped together. This dictionary will be used
# when generalized vectors are used for embedding. Here is meaning of numbers:
# 1: Letters, 2: Marks, 3: Digits, 4: Separators, 5: Punctuations, 6: Symbols, 7: Others
CHAR_TYPE_TO_BUCKET = {
     UCharCategory.UPPERCASE_LETTER: 1,
     UCharCategory.LOWERCASE_LETTER: 1,
     UCharCategory.TITLECASE_LETTER: 1,
     UCharCategory.MODIFIER_LETTER: 1,
     UCharCategory.OTHER_LETTER: 1,
     UCharCategory.NON_SPACING_MARK: 2,
     UCharCategory.ENCLOSING_MARK: 2,
     UCharCategory.COMBINING_SPACING_MARK: 2,
     UCharCategory.DECIMAL_DIGIT_NUMBER: 3,
     # LETTER_NUMBER and OTHER_NUMBER are more like symbols than digits
     UCharCategory.LETTER_NUMBER: 6,
     UCharCategory.OTHER_NUMBER: 6,
     UCharCategory.SPACE_SEPARATOR: 4,
     UCharCategory.LINE_SEPARATOR: 4,
     UCharCategory.PARAGRAPH_SEPARATOR: 4,
     UCharCategory.DASH_PUNCTUATION: 5,
     UCharCategory.START_PUNCTUATION: 5,
     UCharCategory.END_PUNCTUATION: 5,
     UCharCategory.CONNECTOR_PUNCTUATION: 5,
     UCharCategory.OTHER_PUNCTUATION: 5,
     UCharCategory.INITIAL_PUNCTUATION: 5,
     UCharCategory.FINAL_PUNCTUATION: 5,
     UCharCategory.MATH_SYMBOL: 6,
     UCharCategory.CURRENCY_SYMBOL: 6,
     UCharCategory.MODIFIER_SYMBOL: 6,
     UCharCategory.OTHER_SYMBOL: 6,
     UCharCategory.CONTROL_CHAR: 7,
     UCharCategory.FORMAT_CHAR: 7,
     UCharCategory.PRIVATE_USE_CHAR: 7,
     UCharCategory.SURROGATE: 7,
     0: 7
 }
