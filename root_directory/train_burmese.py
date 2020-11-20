from pathlib import Path
from lstm_word_segmentation.lstm_bayesian_optimization import LSTMBayesianOptimization
from tensorflow import keras
from lstm_word_segmentation.text_helpers import divide_train_test_data
from lstm_word_segmentation.word_segmenter import WordSegmenter

# Dividing the my.txt data to train, validation, and test data sets.
'''
divide_train_test_data(input_text=Path.joinpath(Path(__file__).parent.absolute(), 'Data/my.txt'),
                       train_text=Path.joinpath(Path(__file__).parent.absolute(), 'Data/my_train.txt'),
                       valid_text=Path.joinpath(Path(__file__).parent.absolute(), 'Data/my_valid.txt'),
                       test_text=Path.joinpath(Path(__file__).parent.absolute(), 'Data/my_test.txt'),
                       line_limit=12000)
'''

# Use Bayesian optimization to decide on values of hunits and embedding_dim
'''
bayes_optimization = LSTMBayesianOptimization(input_language="Burmese", input_epochs=1,
                                              input_embedding_type='grapheme_clusters_tf', input_clusters_num=350,
                                              input_hunits_lower=4, input_hunits_upper=64, input_embedding_dim_lower=4,
                                              input_embedding_dim_upper=64, input_C=0.05, input_iterations=3)
bayes_optimization.perform_bayesian_optimization()
'''

# Train a new model -- choose name cautiously to not overwrite other models
# '''
model_name = "Burmese_temp_genvec"
word_segmenter = WordSegmenter(input_name=model_name, input_n=50, input_t=100000, input_clusters_num=350,
                               input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=10, input_training_data="my", input_evaluating_data="my",
                               input_language="Burmese", input_embedding_type="generalized_vectors")

# Training, testing, and saving the model
word_segmenter.train_model()
word_segmenter.test_model()
word_segmenter.test_model_line_by_line()
word_segmenter.save_model()
# '''

# Choose one of the saved models to use
'''
model_name = "Burmese_temp"
file = Path.joinpath(Path(__file__).parent.absolute(), 'Models/' + model_name)
model = keras.models.load_model(file)
input_clusters_num = model.weights[0].shape[0]
input_embedding_dim = model.weights[0].shape[1]
input_hunits = model.weights[1].shape[1]//4
if "heavy" in model_name:
    input_n = 200
    input_t = 600000
else:
    input_n = 50
    input_t = 100000

word_segmenter = WordSegmenter(input_name=model_name, input_n=input_n, input_t=input_t,
                               input_clusters_num=input_clusters_num, input_embedding_dim=input_embedding_dim,
                               input_hunits=input_hunits, input_dropout_rate=0.2, input_output_dim=4, input_epochs=15,
                               input_training_data="my", input_evaluating_data="my", input_language="Burmese",
                               input_embedding_type="grapheme_clusters_tf")
word_segmenter.set_model(model)
word_segmenter.test_model()
word_segmenter.test_model_line_by_line()
'''
