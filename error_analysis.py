from pathlib import Path
from tensorflow import keras
from lstm_word_segmentation.lstm_bayesian_optimization import LSTMBayesianOptimization
from lstm_word_segmentation.word_segmenter import WordSegmenter
from lstm_word_segmentation.text_helpers import get_lines_of_text

# Choose one of the saved models to use
# '''
# Thai_model1: Bi-directional LSTM (trained on BEST), grid search
#     thrsh = 350, embedding_dim = 40, hunits = 40
# Thai_model2: Bi-directional LSTM (trained on BEST), grid search + manual reduction of hunits and embedding_size
#     thrsh = 350, embedding_dim = 20, hunits = 20
# Thai_model3: Bi-directional LSTM (trained on BEST), grid search + extreme man reduction of hunits and embedding_size
#     thrsh = 350, embedding_dim = 15, hunits = 15
# Thai_model4: Bi-directional LSTM (trained on BEST), short BayesOpt choice for hunits and embedding_size
#     thrsh = 350, embedding_dim = 16, hunits = 23
# Thai_model5: Bi-directional LSTM (trained on BEST), A very parsimonious model
#     thrsh = 250, embedding_dim = 10, hunits = 10
# Thai_temp: a temporary model, it should be used for storing new models

# For some models the heavy trained versions can be used by adding "_heavy" to the end of the model name. Such as
# Thai_model4_heavy. In training these models n and t are set to 200 and 600000 respectively.

# '''
model_name = "Burmese_model1"
file = Path.joinpath(Path(__file__).parent.absolute(), 'Models/' + model_name)
model = keras.models.load_model(file)
input_clusters_num = model.weights[0].shape[0]
input_embedding_dim = model.weights[0].shape[1]
input_hunits = model.weights[1].shape[1]//4
if "heavy" in model_name:
    input_n = 200
    input_t = 600000
elif "heavier" in model_name:
    input_n = 200
    input_t = 2000000
else:
    input_n = 50
    input_t = 100000

word_segmenter1 = WordSegmenter(input_name=model_name, input_n=input_n, input_t=input_t,
                               input_clusters_num=input_clusters_num, input_embedding_dim=input_embedding_dim,
                               input_hunits=input_hunits, input_dropout_rate=0.2, input_output_dim=4, input_epochs=15,
                               input_training_data="my", input_evaluating_data="my", input_language="Burmese",
                               input_embedding_type="grapheme_clusters_tf")
word_segmenter1.set_model(model)
# word_segmenter.test_model()
# word_segmenter1.test_model_line_by_line()
# '''


# '''
model_name = "Burmese_genvec_12d0"
file = Path.joinpath(Path(__file__).parent.absolute(), 'Models/' + model_name)
model = keras.models.load_model(file)
input_clusters_num = model.weights[0].shape[0]
input_embedding_dim = model.weights[0].shape[1]
input_hunits = model.weights[1].shape[1]//4
if "heavy" in model_name:
    input_n = 200
    input_t = 600000
elif "heavier" in model_name:
    input_n = 200
    input_t = 2000000
else:
    input_n = 50
    input_t = 100000

word_segmenter2 = WordSegmenter(input_name=model_name, input_n=input_n, input_t=input_t,
                               input_clusters_num=input_clusters_num, input_embedding_dim=input_embedding_dim,
                               input_hunits=input_hunits, input_dropout_rate=0.2, input_output_dim=4, input_epochs=15,
                               input_training_data="my", input_evaluating_data="my", input_language="Burmese",
                               input_embedding_type="generalized_vectors")
word_segmenter2.set_model(model)
# word_segmenter2.test_model_line_by_line()

# Testing the model by arbitrary sentences
file = Path.joinpath(Path(__file__).parent.absolute(), 'Data/my_test_segmented.txt')
lines = get_lines_of_text(file, "man_segmented")

cnt = 0
for line in lines[:10]:
    print(line.unsegmented)
    print(line.man_segmented)
    # word_segmenter1.segment_arbitrary_line(line.unsegmented)
    # word_segmenter2.segment_arbitrary_line(line.unsegmented)
    line = "လူမျိုးနွယ်အားဖြင့်"
    word_segmenter1.segment_arbitrary_line(line)
    word_segmenter2.segment_arbitrary_line(line)
    x = input()
# '''