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
'''
model_name = "Burmese_temp"
word_segmenter = WordSegmenter(input_name=model_name, input_n=200, input_t=600000, input_clusters_num=350,
                               input_embedding_dim=28, input_hunits=14, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=20, input_training_data="exclusive my", input_evaluation_data="exclusive my",
                               input_language="Burmese", input_embedding_type="grapheme_clusters_tf")
word_segmenter.train_model()
word_segmenter.save_model()
word_segmenter.test_model_line_by_line(verbose=True)
'''

# Choose one of the saved models to use
# '''
word_segmenter = pick_lstm_model(model_name="Burmese_graphclust_model5_heavy", embedding="grapheme_clusters_tf",
                                 train_data="my", eval_data="my")
print("model_name = {}, embedding dim = {}, hunits = {}".format(word_segmenter.name, word_segmenter.embedding_dim,
                                                                word_segmenter.hunits))
# word_segmenter.save_model()
word_segmenter.test_model_line_by_line(verbose=True, fast=True)
# '''
