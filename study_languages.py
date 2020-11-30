from pathlib import Path
from lstm_word_segmentation.preprocess import evaluate_existing_algorithms, preprocess_thai, preprocess_burmese
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

# Preprocessing Thai and Burmese and making the grapheme clusters dictionaries
'''
preprocess_thai(verbose=False)
print_grapheme_clusters(thrsh=0.99, language="Thai")
preprocess_burmese(verbose=False)
print_grapheme_clusters(thrsh=0.99, language="Burmese")
'''

print_grapheme_clusters(thrsh=0.99, language="Thai-Burmese")