from pathlib import Path
from tensorflow import keras
from lstm_word_segmentation.lstm_bayesian_optimization import LSTMBayesianOptimization
from lstm_word_segmentation.word_segmenter import WordSegmenter
from lstm_word_segmentation.text_helpers import break_lines_based_on_spaces


# Making a version of BEST data where each line is divided according to spaces
'''
input_texts = []
category = ["news", "encyclopedia", "article", "novel"]
for text_num in range(1, 6):
    for cat in category:
        text_num_str = "{}".format(text_num).zfill(5)
        file = Path.joinpath(Path(__file__).parent.absolute(), "Data/Best/{}/{}_".format(cat, cat) +
                             text_num_str + ".txt")
        input_texts.append(file)
output_text = file = Path.joinpath(Path(__file__).parent.absolute(), "Data/Best_spaced.txt")
break_lines_based_on_spaces(input_texts=input_texts, output_text=output_text)
'''

# Use Bayesian optimization to decide on values of hunits and embedding_dim
'''
bayes_optimization = LSTMBayesianOptimization(input_language="Thai", input_epochs=1,
                                              input_embedding_type='grapheme_clusters_tf', input_clusters_num=350,
                                              input_hunits_lower=4, input_hunits_upper=64, input_embedding_dim_lower=4,
                                              input_embedding_dim_upper=64, input_C=0.05, input_iterations=3)
bayes_optimization.perform_bayesian_optimization()
'''

# Train a new model -- choose name cautiously to not overwrite other models
'''
model_name = "Thai_temp_2"
word_segmenter = WordSegmenter(input_name=model_name, input_n=50, input_t=100000, input_clusters_num=350,
                               input_embedding_dim=8, input_hunits=8, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=1, input_training_data="BEST", input_evaluating_data="BEST",
                               input_language="Thai", input_embedding_type="grapheme_clusters_tf")

# Training, testing, and saving the model
word_segmenter.train_model()
# word_segmenter.test_model()
# word_segmenter.test_model_line_by_line()
word_segmenter.save_model()
'''

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

model_name = "Thai_model4_heavy"
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

print(input_embedding_dim)
print(input_hunits)

word_segmenter = WordSegmenter(input_name=model_name, input_n=input_n, input_t=input_t,
                               input_clusters_num=input_clusters_num, input_embedding_dim=input_embedding_dim,
                               input_hunits=input_hunits, input_dropout_rate=0.2, input_output_dim=4, input_epochs=15,
                               input_training_data="BEST", input_evaluating_data="BEST_spaced", input_language="Thai",
                               input_embedding_type=input_embedding_type)
word_segmenter.set_model(model)
# word_segmenter.test_model()
word_segmenter.test_model_line_by_line()

# Testing the model by arbitrary sentences
# line = "แม้จะกะเวลาเอาไว้แม่นยำว่ากว่าเขาจะมาถึงก็คงประมาณหกโมงเย็น"
line = "ทำสิ่งต่างๆ ได้มากขึ้นขณะที่อุปกรณ์ล็อกและชาร์จอยู่ด้วยโหมดแอมเบียนท์"
# line = "เกี่ยวกับนอมินีหรือการถือหุ้นแทนกันในบริษัทต่างๆที่เกี่ยวข้องกับการซื้อหุ้น"
word_segmenter.segment_arbitrary_line(line)
# '''