from pathlib import Path
from lstm_word_segmentation.preprocess import evaluate_existing_algorithms, preprocess_thai, preprocess_burmese, \
                                              make_thai_burmese_dictionary
from lstm_word_segmentation.helpers import print_grapheme_clusters
from lstm_word_segmentation.text_helpers import compute_accuracy


# Evaluate ICU on Burmese
'''
file = Path.joinpath(Path(__file__).parent.absolute(), 'Data/my_test_segmented.txt')
acc = compute_accuracy(file=file, segmentation_type="icu")
print(acc.get_bies_accuracy())
print(acc.get_f1_score())
'''

# Evaluate ICU and Deepcut for Thai
# evaluate_existing_algorithms()

# Evaluate ICU on spaced BEST data for for Thai
'''
file = Path.joinpath(Path(__file__).parent.absolute(), 'Data/Best_spaced_test.txt')
acc = compute_accuracy(file=file, segmentation_type="icu")
print(acc.get_bies_accuracy())
print(acc.get_f1_score())
'''

# Preprocessing Thai and Burmese and making the grapheme clusters dictionaries
# '''
preprocess_thai(verbose=False, exclusive=True)
print_grapheme_clusters(thrsh=0.99, language="Thai")
# preprocess_burmese(verbose=False)
# print_grapheme_clusters(thrsh=0.99, language="Burmese")
# make_thai_burmese_dictionary()
# print_grapheme_clusters(thrsh=0.99, language="Thai-Burmese")
# '''

