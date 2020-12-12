from pathlib import Path
from lstm_word_segmentation.preprocess import evaluate_existing_algorithms, find_grapheme_clusters, \
                                              make_thai_burmese_dictionary
from lstm_word_segmentation.helpers import print_grapheme_clusters
from lstm_word_segmentation.text_helpers import make_thai_specific_best_data, permute_lines_of_text, \
                                                divide_train_test_data


# Making exclusive BEST data set: a version of BEST data that has only Thai-script code points.
# make_thai_specific_best_data()

# Dividing the my.txt data to train, validation, and test data sets.
'''
permute_lines_of_text('Data/my.txt', 'Data/my_permutated.txt')
divide_train_test_data(input_text=Path.joinpath(Path(__file__).parent.absolute(), 'Data/my_permutated.txt'),
                       train_text=Path.joinpath(Path(__file__).parent.absolute(), 'Data/my_train.txt'),
                       valid_text=Path.joinpath(Path(__file__).parent.absolute(), 'Data/my_valid.txt'),
                       test_text=Path.joinpath(Path(__file__).parent.absolute(), 'Data/my_test.txt'),
                       line_limit=21000)
'''


# Evaluate ICU and Deepcut for Thai
# '''
evaluate_existing_algorithms(algorithm="ICU", data="BEST")
evaluate_existing_algorithms(algorithm="ICU", data="exclusive BEST")
evaluate_existing_algorithms(algorithm="Deepcut", data="BEST")
evaluate_existing_algorithms(algorithm="ICU", data="SAFT Burmese")
# '''

# Using BEST data set and Google Corpus crawler to make grapheme clusters for Thai and Burmese
'''
# Thai
find_grapheme_clusters(language="Thai", exclusive=True, verbose=False)
print_grapheme_clusters(thrsh=0.99, language="Thai")

# Burmese
find_grapheme_clusters(language="Burmese", exclusive=True, verbose=False)
print_grapheme_clusters(thrsh=0.99, language="Burmese")

# Multilingual Thai/Burmese
make_thai_burmese_dictionary()
print_grapheme_clusters(thrsh=0.99, language="Thai-Burmese")
'''
