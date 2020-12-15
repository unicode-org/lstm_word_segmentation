from lstm_word_segmentation.lstm_bayesian_optimization import LSTMBayesianOptimization
from lstm_word_segmentation.word_segmenter import pick_lstm_model
from lstm_word_segmentation.word_segmenter import WordSegmenter


# Use Bayesian optimization to decide on values of hunits and embedding_dim
'''
bayes_optimization = LSTMBayesianOptimization(input_language="Burmese", input_n=50, input_t=10000, input_epochs=2,
                                              input_embedding_type='grapheme_clusters_tf', input_clusters_num=350,
                                              input_training_data="my", input_evaluation_data="my",
                                              input_hunits_lower=4, input_hunits_upper=64, input_embedding_dim_lower=4,
                                              input_embedding_dim_upper=64, input_c=0.05, input_iterations=1)
bayes_optimization.perform_bayesian_optimization()
'''

# Train a new model -- choose name cautiously to not overwrite other models
# '''
model_name = "Burmese_codepoints_exclusive_model4_heavy_"
word_segmenter = WordSegmenter(input_name=model_name, input_n=50, input_t=100000, input_clusters_num=350,
                               input_embedding_dim=28, input_hunits=14, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=20, input_training_data="exclusive my", input_evaluation_data="exclusive my",
                               input_language="Burmese", input_embedding_type="codepoints")
word_segmenter.train_model()
word_segmenter.save_model()
word_segmenter.test_model_line_by_line(verbose=True)
# '''

# Choose one of the saved models to use
'''
word_segmenter = pick_lstm_model(model_name="Burmese_codepoints_exclusive_model4_heavy_", embedding="codepoints",
                                 train_data="exclusive my", eval_data="exclusive my")
print("embedding dim = {}".format(word_segmenter.embedding_dim))
print("hunits = {}".format(word_segmenter.hunits))
word_segmenter.save_model()
word_segmenter.test_model_line_by_line(verbose=True, fast=True)
'''
