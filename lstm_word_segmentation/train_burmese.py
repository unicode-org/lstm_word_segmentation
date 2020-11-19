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

################################ Processing Burmese ################################
# Testing how ICU detects grapheme clusters and how it segments Burmese (will be deleted later on)
'''
str = "ြင်သစ်မှာ နောက်လလုပ်မယ့် သမ္မတရွေးကောက်ပွဲမှာ သူဝင်ပြိုင်မှာ မဟုတ်ဘူးလို့ ဝန်ကြီးချုပ်ဟောင်း အလိန်ယူပေက ကြေညာလိုက်ပါတယ်။"
chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getRoot())
word_break_iterator = BreakIterator.createWordInstance(Locale.getRoot())
chars_break_iterator.setText(str)
word_break_iterator.setText(str)
char_brkpoints = [0]
for brkpoint in chars_break_iterator:
    char_brkpoints.append(brkpoint)
word_brkpoints = [0]
for brkpoint in word_break_iterator:
    word_brkpoints.append(brkpoint)
print(char_brkpoints)
print(word_brkpoints)
x = input()
'''

# Preprocess the Burmese language
# Burmese_graph_clust_ratio = preprocess_burmese(verbose=False)
# np.save(os.getcwd() + '/Data/Burmese_graph_clust_ratio.npy', Burmese_graph_clust_ratio)

# Loading the graph_clust from memory
graph_clust_ratio = np.load(os.getcwd() + '/Data/Burmese_graph_clust_ratio.npy', allow_pickle=True).item()
# print_grapheme_clusters(ratios=graph_clust_ratio, thrsh=0.99)

# Dividing the my.txt data to train, validation, and test data sets.
# divide_train_test_data(input_text="./Data/my.txt", train_text="./Data/my_train.txt", valid_text="./Data/my_valid.txt",
#                        test_text="./Data/my_test.txt")
# Making a ICU segmented version of the test data, for future tests
# store_icu_segmented_file(unseg_filename="./Data/my_test.txt", seg_filename="./Data/my_test_segmented.txt")

# Train a new model -- choose name cautiously to not overwrite other models
'''
model_name = "Burmese_temp"
cnt = 0
graph_thrsh = 350  # The vocabulary size for embeddings
graph_clust_dic = dict()
for key in graph_clust_ratio.keys():
    if cnt < graph_thrsh-1:
        graph_clust_dic[key] = cnt
    if cnt == graph_thrsh-1:
        break
    cnt += 1

word_segmenter = WordSegmenter(input_n=50, input_t=100000, input_graph_clust_dic=graph_clust_dic,
                               input_embedding_dim=20, input_hunits=20, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=3, input_training_data="my", input_evaluating_data="my")

# Training and saving the model
word_segmenter.train_model()
word_segmenter.test_model()
fitted_model = word_segmenter.get_model()
fitted_model.save("./Models/" + model_name)
np.save(os.getcwd() + "/Models/" + model_name + "/" + "weights", fitted_model.weights)
# Saving the model in json format to be used later by rust code
write_json(model_name, fitted_model)
'''

# Choose one of the saved models to use
'''
model_name = "Burmese_temp"
model = keras.models.load_model("./Models/" + model_name)
input_graph_thrsh = model.weights[0].shape[0]
input_embedding_dim = model.weights[0].shape[1]
input_hunits = model.weights[1].shape[1]//4

# Building the model instance and loading the trained model
cnt = 0
graph_thrsh = input_graph_thrsh  # The vocabulary size for embeddings
graph_clust_dic = dict()
for key in graph_clust_ratio.keys():
    if cnt < graph_thrsh-1:
        graph_clust_dic[key] = cnt
    if cnt == graph_thrsh-1:
        break
    cnt += 1
word_segmenter = WordSegmenter(input_n=50, input_t=100000, input_graph_clust_dic=graph_clust_dic,
                               input_embedding_dim=input_embedding_dim, input_hunits=input_hunits,
                               input_dropout_rate=0.2, input_output_dim=4, input_epochs=3,
                               input_training_data="my", input_evaluating_data="my")
word_segmenter.set_model(model)

# Testing the model
word_segmenter.test_model()
word_segmenter.test_model_line_by_line()
'''
