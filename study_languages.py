from lstm_word_segmentation.preprocess import evaluate_existing_algorithms, preprocess_thai, preprocess_burmese
from lstm_word_segmentation.helpers import print_grapheme_clusters


evaluate_existing_algorithms()
preprocess_thai(verbose=False)
print_grapheme_clusters(thrsh=0.99, language="Thai")
preprocess_burmese(verbose=False)
print_grapheme_clusters(thrsh=0.99, language="Burmese")
