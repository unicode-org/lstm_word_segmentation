import numpy as np
from pathlib import Path
from icu import UCharCategory
from collections import Counter

# The dictionary that stores different grapheme clusters of Thai and the ratio that each appear in Thai texts. It is
# computed using BEST dataset
# path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/Data/Thai_graph_clust_ratio.npy"

# path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Data/Thai_graph_clust_ratio.npy'
path = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/Thai_graph_clust_ratio.npy')
THAI_GRAPH_CLUST_RATIO = np.load(str(path), allow_pickle=True).item()

path = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/Thai_graph_clust_ratio.npy')
THAI_EXCLUSIVE_GRAPH_CLUST_RATIO = np.load(str(path), allow_pickle=True).item()

# The dictionary that stores different grapheme clusters of Burmese and the ratio that each appear in Burmese texts.
path = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/Burmese_graph_clust_ratio.npy')
BURMESE_GRAPH_CLUST_RATIO = np.load(str(path), allow_pickle=True).item()

# The dictionary that stores different grapheme clusters of Burmese and Thaiand the ratio that each appear.
path = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/Thai_Burmese_graph_clust_ratio.npy')
THAI_BURMESE_GRAPH_CLUST_RATIO = np.load(str(path), allow_pickle=True).item()


# A dictionary that determines how different types of code points are grouped together. This dictionary will be used
# when generalized vectors are used for embedding.
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
