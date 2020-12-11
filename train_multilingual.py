from pathlib import Path
from tensorflow import keras
from lstm_word_segmentation.lstm_bayesian_optimization import LSTMBayesianOptimization
from lstm_word_segmentation.word_segmenter import WordSegmenter
from lstm_word_segmentation.text_helpers import merge_two_texts, add_additional_bars


# Making new files with bars around spaces in my_train.txt and my_valid.txt, so it is compatible with the format of BEST
# data sets.
# '''
input_file = Path.joinpath(Path(__file__).parent.absolute(), "Data/my_train.txt")
output_file = Path.joinpath(Path(__file__).parent.absolute(), "Data/my_train_with_bars.txt")
add_additional_bars(read_filename=input_file, write_filename=output_file)

input_file = Path.joinpath(Path(__file__).parent.absolute(), "Data/my_valid.txt")
output_file = Path.joinpath(Path(__file__).parent.absolute(), "Data/my_valid_with_bars.txt")
add_additional_bars(read_filename=input_file, write_filename=output_file)
# '''

# Making a dataset that has both Thai (BEST) and Burmese (my) in it
# '''
thai_texts = []
category = ["news", "encyclopedia", "article", "novel"]
for text_num in range(10, 20):
    for cat in category:
        text_num_str = "{}".format(text_num).zfill(5)
        file = Path.joinpath(Path(__file__).parent.absolute(), "Data/Best/{}/{}_".format(cat, cat) +
                             text_num_str + ".txt")
        thai_texts.append(file)
burmese_texts = [Path.joinpath(Path(__file__).parent.absolute(), "Data/my_valid_with_bars.txt")]
output_text = file = Path.joinpath(Path(__file__).parent.absolute(), "Data/Best_my_valid.txt")
merge_two_texts(input_texts1=thai_texts, input_texts2=burmese_texts, output_text=output_text, line_limit=1200)
# '''

# Train a new model -- choose name cautiously to not overwrite other models
# '''
model_name = "Thai_Burmese_temp"
word_segmenter = WordSegmenter(input_name=model_name, input_n=100, input_t=200000, input_clusters_num=500,
                               input_embedding_dim=40, input_hunits=40, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=15, input_training_data="BEST_my", input_evaluation_data="BEST_my",
                               input_language="Thai_Burmese", input_embedding_type="grapheme_clusters_tf")

# Training, testing, and saving the model
word_segmenter.train_model()
word_segmenter.save_model()
# '''

# Choose one of the saved models to use
model_name = "Thai_Burmese_temp"
input_embedding_type = "grapheme_clusters_tf"
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

print("embedding dim = {}".format(input_embedding_dim))
print("hunits = {}".format(input_hunits))

word_segmenter = WordSegmenter(input_name=model_name, input_n=input_n, input_t=input_t,
                               input_clusters_num=input_clusters_num, input_embedding_dim=input_embedding_dim,
                               input_hunits=input_hunits, input_dropout_rate=0.2, input_output_dim=4, input_epochs=15,
                               input_training_data="BEST_my", input_evaluation_data="BEST_my",
                               input_language="Thai_Burmese", input_embedding_type=input_embedding_type)
word_segmenter.set_model(model)
word_segmenter.test_model_line_by_line(verbose=True)
