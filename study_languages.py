from lstm_word_segmentation.preprocess import preprocess_saft_data, preprocess_thai, preprocess_burmese
from lstm_word_segmentation.helpers import print_grapheme_clusters

from collections import Counter
from lstm_word_segmentation import constants
from icu import Char
smallest_unicode_dec = int("1000", 16)
largest_unicode_dec = int("109F", 16)
# smallest_unicode_dec = int("0E01", 16)
# largest_unicode_dec = int("0E5B", 16)
dic = Counter()
for i in range(smallest_unicode_dec, largest_unicode_dec + 1):
    ch = chr(i)
    dic[constants.CHAR_TYPE_TO_BUCKET[Char.charType(ch)]] += 1
    if constants.CHAR_TYPE_TO_BUCKET[Char.charType(ch)] == 3:
        print(ch)
print(dic)
x = input()

preprocess_saft_data()
preprocess_thai(verbose=False)
print_grapheme_clusters(thrsh=0.99, language="Thai")
preprocess_burmese(verbose=False)
print_grapheme_clusters(thrsh=0.99, language="Burmese")
