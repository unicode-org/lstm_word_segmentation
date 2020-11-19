# coding=utf-8
import numpy as np
import os
from icu import BreakIterator, Locale, Char, UCharCategory
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf
# from keras import optimizer
from bayes_opt import BayesianOptimization
import json
import pickle
from collections import Counter

import constants
from helpers import is_ascii, diff_strings, sigmoid, clean_line, clean_line
from accuracy import Accuracy
from line import Line
from bies import Bies
from grapheme_cluster import GraphemeCluster
import preprocess
from word_segmenter import WordSegmenter

################################ Processing Thai ################################


# preprocess.preprocess_thai(verbose=False)

# Performing Bayesian optimization to find the best value for hunits and embedding_dim
'''
cnt = 0
graph_thrsh = 350  # The vocabulary size for embeddings
graph_clust_dic = dict()
for key in graph_clust_ratio.keys():
    if cnt < graph_thrsh-1:
        graph_clust_dic[key] = cnt
    if cnt == graph_thrsh-1:
        break
    cnt += 1
perform_bayesian_optimization(hunits_lower=4, hunits_upper=64, embedding_dim_lower=4, embedding_dim_upper=64)
'''

# Train a new model -- choose name cautiously to not overwrite other models
# '''
model_name = "Thai_temp"
word_segmenter = WordSegmenter(input_name=model_name, input_n=50, input_t=10000, input_clusters_num=350,
                               input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=1, input_training_data="BEST", input_evaluating_data="BEST",
                               input_language="Thai", input_embedding_type="grapheme_clusters_tf")

# Training and saving the model
word_segmenter.train_model()
word_segmenter.test_model()
word_segmenter.test_model_line_by_line()
x = input("after fitting")

fitted_model = word_segmenter.get_model()
fitted_model.save("./Models/" + model_name)
np.save(os.getcwd() + "/Models/" + model_name + "/" + "weights", fitted_model.weights)
write_model_json(model_name, graph_clust_dic, fitted_model)
# '''

# Choose one of the saved models to use
'''
# Thai_model1: Bi-directional LSTM (trained on BEST), grid search
#     thrsh = 350, embedding_dim = 40, hunits = 40
# Thai_model2: Bi-directional LSTM (trained on BEST), grid search + manual reduction of hunits and embedding_size
#     thrsh = 350, embedding_dim = 20, hunits = 20
# Thai_model3: Bi-directional LSTM (trained on BEST), grid search + extreme manual reduction of hunits and embedding_size
#     thrsh = 350, embedding_dim = 15, hunits = 15
# Thai_model4: Bi-directional LSTM (trained on BEST), short BayesOpt choice for hunits and embedding_size
#     thrsh = 350, embedding_dim = 16, hunits = 23
# Thai_model5: Bi-directional LSTM (trained on BEST), A very parsimonious model
#     thrsh = 250, embedding_dim = 10, hunits = 10
# Thai_temp: a temporary model, it should be used for storing new models

# For some models the heavy trained versions can be used by adding "_heavy" to the end of the model name. Such as
# Thai_model4_heavy. In training these models n and t are set to 200 and 600000 respectively.

model_name = "Thai_model4"
model = keras.models.load_model("./Models/" + model_name)
input_graph_thrsh = model.weights[0].shape[0]
input_embedding_dim = model.weights[0].shape[1]
input_hunits = model.weights[1].shape[1]//4
if "heavy" in model_name:
    input_n = 200
    input_t = 600000
else:
    input_n = 50
    input_t = 100000

# Building the model instance and loading the trained model
cnt = 0
graph_thrsh = input_graph_thrsh  # The vocabulary size for embeddings
graph_clust_dic = dict()
for key in constants.THAI_GRAPH_CLUST_RATIO.keys():
    if cnt < graph_thrsh-1:
        graph_clust_dic[key] = cnt
    if cnt == graph_thrsh-1:
        break
    cnt += 1

# write_grapheme_clusters_dic_json(graph_clust_ratio, graph_thrsh)
word_segmenter = WordSegmenter(input_n=input_n, input_t=input_t, input_graph_clust_dic=graph_clust_dic,
                               input_embedding_dim=input_embedding_dim, input_hunits=input_hunits,
                               input_dropout_rate=0.2, input_output_dim=4, input_epochs=15,
                               input_training_data="BEST", input_evaluating_data="BEST", input_language="Thai",
                               input_embedding_type="grapheme_clusters_tf")
word_segmenter.set_model(model)


# Testing the model by arbitrary sentences
line = "แม้จะกะเวลาเอาไว้แม่นยำว่ากว่าเขาจะมาถึงก็คงประมาณหกโมงเย็น"
word_segmenter.segment_arbitrary_line(line)
x = input()

# Testing the model using large texts
word_segmenter.test_model()
word_segmenter.test_model_line_by_line()
# '''

# Code for testing the bies normalizer function
# Testing the normalizer
# input_bies = "bbiies"
# print(normalize_bies(input_bies))


