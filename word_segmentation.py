# import PyICU as pyicu
import numpy as np
import os
import icu
import datetime
from icu import UnicodeString, BreakIterator, Locale
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Dropout

# Just in case you needed this code to find the original code that used it
''' 
def print_forward(boundary):
    # Prints each element in forward order
    # Ex. If the boundary is for sentences you will print one sentence at each instance,
    # but if the boundary is for words you will print one word at each instance

    first_bp = boundary.first()
    for second_bp in boundary:
        print_all_text_range(boundary, first_bp, second_bp)
        first_bp = second_bp
'''


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def get_segmented_string(str, brkpoints):
    # Prints the original text segmented
    out = "|"
    for i in range(len(brkpoints)-1):
        out += str[brkpoints[i]: brkpoints[i+1]] + "|"
    return out

def get_bies(char_brkpoints, word_brkpoints):
    bies = np.zeros(shape=[4, len(char_brkpoints)-1])
    word_ind = 0
    for i in range(len(char_brkpoints)-1):
        word_st = word_brkpoints[word_ind]
        word_fn = word_brkpoints[word_ind + 1]
        char_st = char_brkpoints[i]
        char_fn = char_brkpoints[i+1]
        if char_st == word_st and char_fn != word_fn:
            bies[0, i] = 1
            continue
        if char_st != word_st and char_fn != word_fn:
            bies[1, i] = 1
            continue
        if char_st != word_st and char_fn == word_fn:
            bies[2, i] = 1
            word_ind += 1
            continue
        if char_st == word_st and char_fn == word_fn:
            bies[3, i] = 1
            word_ind += 1
            continue
    return bies

def print_bies(str, bies):
    print("BIES:")
    print(str)
    for i in range(bies.shape[1]):
        if bies[0, i] == 1:
            print("b", end="")
        if bies[1, i] == 1:
            print("i", end="")
        if bies[2, i] == 1:
            print("e", end="")
        if bies[3, i] == 1:
            print("s", end="")
    print()

def get_bies_string (bies):
    out = ""
    for i in range(bies.shape[1]):
        if bies[0, i] == 1:
            out += "b"
        if bies[1, i] == 1:
            out += "i"
        if bies[2, i] == 1:
            out += "e"
        if bies[3, i] == 1:
            out += "s"
    return out

def get_bies_string_from_softmax (mat):
    out = ""
    for i in range(mat.shape[0]):
        max_softmax = max(mat[i, :])
        if mat[i, 0] == max_softmax:
            out += "b"
        if mat[i, 1] == max_softmax:
            out += "i"
        if mat[i, 2] == max_softmax:
            out += "e"
        if mat[i, 3] == max_softmax:
            out += "s"
    return out

def remove_tags2(line, st_tag, fn_tag):
    new_line = ""
    st_ind = 0
    while st_ind < len(line):
        if line[st_ind: st_ind+len(st_tag)] == st_tag:
            fn_ind = st_ind
            while fn_ind < len(line):
                if line[fn_ind: fn_ind+len(fn_tag)] == fn_tag:
                    fn_ind = fn_ind+len(fn_tag) + 1
                    # print("st_ind = {}, fn_ind = {}".format(st_ind, fn_ind))
                    if st_ind -2 >= 0 and fn_ind+2 <= len(line):
                        if line[st_ind-2:st_ind] == " |" and line[fn_ind:fn_ind+2] == " |":
                            fn_ind += 2
                            # print("special case")
                    st_ind = fn_ind
                    break
                else:
                    fn_ind += 1
        if st_ind < len(line):
            new_line += line[st_ind]
        st_ind += 1
        # print(st_ind)
    # print(line)
    # print(new_line)
    # x = input()
    return new_line

def remove_tags(line, st_tag, fn_tag):
    st_delete = []
    fn_delete = []
    for i in range(len(line) - len(fn_tag)):
        if line[i:i + len(st_tag)] == st_tag:
            st_delete.append(i)
        if line[i:i + len(fn_tag)] == fn_tag:
            fn_delete.append(i + len(fn_tag))
    if len(st_delete) != len(fn_delete):
        print("WARNING: number of {} and {} are not equal".format(st_tag, fn_tag))


    if len(st_delete) == 0:
        new_line = line
    else:
        new_line = line[0: st_delete[0]]
        for i in range(len(fn_delete) - 1):
            add = 0
            if st_delete[i] == 0 and line[fn_delete[0]+1] == ' ':
                add = 1
            new_line += line[fn_delete[i] + 1 + add: st_delete[i + 1]]
        new_line += line[fn_delete[len(fn_delete) - 1] + 1:]
    new_line = new_line.replace("| | |", "| |")
    return new_line

def get_text_temp(str, times, n):
    # Extracts the training data ready for embedding from str
    X = np.zeros(shape=[times, 1])
    Y = np.zeros(shape=[times, 4])
    for i in range(times):
        new_char = rd.choice(string.ascii_letters)
        new_int = 0
        if ord('A') <= ord(new_char) <= ord('Z'):
            new_int = ord(new_char) - ord('A')
        elif ord('a') <= ord(new_char) <= ord('z'):
            new_int = ord(new_char) - ord('a') + 26
        X[i, 0] = new_int

        if new_char in ['a', 'e', 'o', 'i', 'u', 'A', 'E', 'O', 'I', 'U']:
            Y[i, 0] = 1
            Y[i, 1] = 0
        else:
            Y[i, 0] = 0
            Y[i, 1] = 1
    return X, Y

def get_BEST_text(starting_text, ending_text):
    category = ["news", "encyclopedia", "article", "novel"]
    out_str = ""
    for text_num in range(starting_text, ending_text):
        for cat in category:
            text_num_str = "{}".format(text_num).zfill(5)
            file = open("./Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt", 'r')
            line_counter = 0
            while True:
                line = file.readline().strip()
                if not line:
                    break
                if len(line) >= 1 and line[len(line) - 1] != '|':
                    line += "|"
                line = remove_tags2(line, "<NE>", "</NE>")
                line = remove_tags2(line, "<AB>", "</AB>")
                if "http" in line or len(line) == 0:
                    continue
                # print(line)
                # x = input()
                out_str += line
                line_counter += 1
    if out_str[0] != '|':
        out_str = '|' + out_str
    return out_str

def get_trainable_data(input_line, graph_clust_ids):
    # Finding word breakpoints
    word_brkpoints = []
    found_bars = 0
    for i in range(len(input_line)):
        if input_line[i] == '|':
            word_brkpoints.append(i - found_bars)
            found_bars += 1
    unsegmented_line = input_line.replace("|", "")

    # Finding grapheme cluster breakpoints
    chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getUS())
    chars_break_iterator.setText(unsegmented_line)
    char_brkpoints = [0]
    for brkpoint in chars_break_iterator:
        char_brkpoints.append(brkpoint)

    # Finding bies
    true_bies = get_bies(char_brkpoints, word_brkpoints)
    true_bies_str = get_bies_string(true_bies)

    times = len(char_brkpoints)-1
    x_data = np.zeros(shape=[times, 1])
    y_data = np.zeros(shape=[times, 4])

    excess_grapheme_ids = max(grapheme_clusters_ids.values()) + 1
    for i in range(times):
        char_st = char_brkpoints[i]
        char_fn = char_brkpoints[i + 1]
        curr_char = unsegmented_line[char_st: char_fn]
        x_data[i, 0] = graph_clust_ids.get(curr_char, excess_grapheme_ids)
        y_data[i, :] = true_bies[:, i]
    return x_data, y_data

def get_pseudo_trainable_data(input_str, times, n, graph_clust_ids):
    x_data = np.zeros(shape=[times, 1])
    y_data = np.zeros(shape=[times, 4])
    # Finding word breakpoints
    unsegmented_str = input_str.replace("|", "")
    words_break_iterator = BreakIterator.createWordInstance(Locale.getUS())
    words_break_iterator.setText(unsegmented_str)
    word_brkpoints = [0]
    for brkpoint in words_break_iterator:
        word_brkpoints.append(brkpoint)

    # Finding grapheme cluster breakpoints
    chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getUS())
    chars_break_iterator.setText(unsegmented_str)
    char_brkpoints = [0]
    for brkpoint in chars_break_iterator:
        char_brkpoints.append(brkpoint)

    # Finding bies
    pseudo_bies = get_bies(char_brkpoints, word_brkpoints)
    pseudo_bies_str = get_bies_string(pseudo_bies)

    # Printing for debugging
    print("INPUT STR: {}".format(input_str[:200]))
    print("UNSEG STR: {}".format(unsegmented_str[:200]))
    print("BIES  STR: {}".format(pseudo_bies_str[:200]))

    # Creating x_data and y_data
    excess_grapheme_ids = max(graph_clust_ids.values())
    for i in range(times):
        char_st = char_brkpoints[i]
        char_fn = char_brkpoints[i + 1]
        curr_char = unsegmented_str[char_st: char_fn]
        x_data[i, 0] = graph_clust_ids.get(curr_char, excess_grapheme_ids)
        y_data[i, :] = pseudo_bies[:, i]
    return x_data, y_data

def compute_hc(weight, x_t, h_tm1, c_tm1):
    '''
    c_tm1 = np.array([0,0]).reshape(1,2)
    h_tm1 = np.array([0,0]).reshape(1,2)
    x_t   = np.array([1]).reshape(1,1)

    warr.shape = (nfeature, hunits*4)
    uarr.shape = (hunits, hunits*4)
    barr.shape = (hunits*4, )
    '''

    warr, uarr, barr = weight
    warr = warr.numpy()
    uarr = uarr.numpy()
    barr = barr.numpy()

    s_t = (x_t.dot(warr) + h_tm1.dot(uarr) + barr)
    hunit = uarr.shape[0]
    i = sigmoid(s_t[:, :hunit])
    f = sigmoid(s_t[:, 1 * hunit:2 * hunit])
    _c = np.tanh(s_t[:, 2 * hunit:3 * hunit])
    o = sigmoid(s_t[:, 3 * hunit:])
    c_t = i * _c + f * c_tm1
    h_t = o * np.tanh(c_t)
    return [h_t, c_t]

class KerasBatchGenerator(object):
    def __init__(self, x_data, y_data, time_steps, batch_size, dim_features, dim_output, times):
        self.x_data = x_data  # dim = times * dim_features
        self.y_data = y_data  # dim = times * dim_output
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.dim_features = dim_features
        self.dim_output = dim_output
        self.times = times

    def generate(self):
        x = np.zeros([self.batch_size, self.time_steps])
        y = np.zeros([self.batch_size, self.time_steps, self.dim_output])
        while True:
            for i in range(self.batch_size):
                x[i, :] = self.x_data[self.time_steps * i: self.time_steps * (i + 1), 0]
                y[i, :, :] = self.y_data[self.time_steps * i: self.time_steps * (i + 1), :]
            yield x, y

    def generate_all_batches(self):
        x = np.zeros([self.batch_size, self.time_steps])
        y = np.zeros([self.batch_size, self.time_steps, self.dim_output])
        for i in range(self.batch_size):
            x[i, :] = self.x_data[self.time_steps * i: self.time_steps * (i + 1), 0]
            y[i, :, :] = self.y_data[self.time_steps * i: self.time_steps * (i + 1), :]
        return x, y


class WordSegmenter:
    def __init__(self, input_n, input_t, input_graph_clust_dic, input_embedding_dim, input_hunits, input_dropout_rate, input_feature_dim,
                 input_output_dim, input_epochs, input_training_data, input_evaluating_data):
        # n: default length of the input for LSTM model
        # T: total length of data used to train and validate the model
        # batch_num: number of batches used to train the model
        # graph_clust_dic: a dictionary that matches most used grapheme clusters to integers
        # clusters_num: number of grapheme clusters used in graph_clust_dic
        # embedding_dim: length of the embedding vectors for each grapheme cluster
        # hunits: number of units used in each cell of LSTM
        # dropout_rate: dropout rate usd in after bidirectional LSTM layer
        # feature_dim: dimension of the input layer
        # output_dim: dimension of the output layer
        # epochs: number of epochs used to train the model
        # training_data: name of the data used to train the model
        # evaluating_data: name of the data used to evaluate the model

        self.n = input_n
        self.t = input_t
        if self.t % self.n != 0:
            print("Warning: t is not divided by n")
        self.batch_num = self.t // self.n
        self.graph_clust_dic = input_graph_clust_dic
        self.clusters_num = len(self.graph_clust_dic.keys()) + 1
        self.embedding_dim = input_embedding_dim
        self.hunits = input_hunits
        self.dropout_rate = input_dropout_rate
        self.feature_dim = input_feature_dim
        self.output_dim = input_output_dim
        self.epochs = input_epochs
        self.training_data = input_training_data
        self.evaluating_data = input_evaluating_data

    def train_model(self):

        # Get training data of length self.t
        if self.training_data == "BEST":
            # this chunk of data has ~ 10^6 data points
            input_str = get_BEST_text(starting_text=1, ending_text=6)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        else:
            print("Warning: no implementation for this training data exists!")

        train_generator = KerasBatchGenerator(x_data, y_data, time_steps=self.n, batch_size=self.batch_num,
                                              dim_features=self.feature_dim, dim_output=self.output_dim, times=self.t)

        # Get validation data of length self.t
        if self.training_data == "BEST":
            # this chunk of data has ~ 10^6 data points
            input_str = get_BEST_text(starting_text=10, ending_text=16)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        else:
            print("Warning: no implementation for this validation data exists!")
        valid_generator = KerasBatchGenerator(x_data, y_data, time_steps=self.n, batch_size=self.batch_num,
                                              dim_features=self.feature_dim, dim_output=self.output_dim, times=self.t)

        # Building the model
        model = Sequential()
        model.add(Embedding(self.clusters_num, self.embedding_dim, input_length=self.n))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(self.hunits, return_sequences=True), input_shape=(self.n, 1)))
        model.add(Dropout(self.dropout_rate))
        model.add(TimeDistributed(Dense(self.output_dim, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fitting the model
        model.fit(train_generator.generate(), steps_per_epoch=self.t//self.batch_num,
                  epochs=self.epochs, validation_data=valid_generator.generate(),
                  validation_steps=self.t//self.batch_num)
        self.model = model

    def test_model(self):
        # Get test data
        if self.evaluating_data == "BEST":
            # this chunk of data has ~ 10^6 data points
            input_str = get_BEST_text(starting_text=30, ending_text=36)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        else:
            print("Warning: no implementation for this evaluation data exists!")
        test_generator = KerasBatchGenerator(x_data, y_data, time_steps=self.n, batch_size=self.batch_num,
                                              dim_features=self.feature_dim, dim_output=self.output_dim, times=self.t)

        # Testing batch by batch (each batch of length n)
        all_test_input, all_actual_y = test_generator.generate_all_batches()
        all_y_hat = self.model.predict(all_test_input)
        test_acc = []
        for i in range(self.t // self.n):  # for each batch
            actual_y = all_actual_y[i, :, :]
            actual_y = get_bies_string(np.transpose(actual_y))
            y_hat = all_y_hat[i, :, :]
            y_hat = get_bies_string_from_softmax(y_hat)
            mismatch = 0
            for j in range(len(actual_y)):
                if actual_y[j] != y_hat[j]:
                    mismatch += 1
            test_acc.append(1 - mismatch / len(actual_y))
        test_acc = np.array(test_acc)
        print("the average test accuracy: {}".format(np.mean(test_acc)))

    def test_text(self, cat, text_num):
        text_num_str = "{}".format(text_num).zfill(5)
        file = open("./Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt", 'r')
        test_acc = []
        while True:
            line = file.readline().strip()
            if not line:
                break
            if len(line) >= 1 and line[len(line) - 1] != '|':
                line += "|"
            line = remove_tags2(line, "<NE>", "</NE>")
            line = remove_tags2(line, "<AB>", "</AB>")
            if "http" in line or len(line) == 0:
                continue
            if line[0] != '|':
                line = '|' + line
            x_data, y_data = get_trainable_data(line, self.graph_clust_dic)
            y_hat = self.model.predict(x_data)
            y_hat = y_hat[:, 0, :]
            y_hat = get_bies_string_from_softmax(y_hat)
            actual_y = get_bies_string_from_softmax(y_data)
            mismatch = 0
            for j in range(len(actual_y)):
                if actual_y[j] != y_hat[j]:
                    mismatch += 1
            test_acc.append(1 - mismatch / len(actual_y))
            # print(actual_y)
            # print(y_hat)
        print("the average test accuracy: {}".format(np.mean(test_acc)))
        return test_acc

    def test_model_line_by_line(self):
        all_test_acc = []
        category = ["news", "encyclopedia", "article", "novel"]
        for text_num in range(30, 36):
            for cat in category:
                all_test_acc += self.test_text(cat, text_num)
        print("the average test accuracy: {}".format(np.mean(test_acc)))

# Pre-processing for grapheme clusters analysis and icu accuracy computations
'''
grapheme_clusters_dic = dict()
icu_mismatch = 0
icu_total_bies_lengths = 0
for text_num in range(1, 96):
    # Computing the performance of the current algorithm
    text_num_str = "{}".format(text_num).zfill(5)
    print("TEXT {}".format(text_num))
    # file = open("./Data/Best/news/news_" + text_num_str + ".txt", 'r')
    # file = open("./Data/Best/encyclopedia/encyclopedia_" + text_num_str + ".txt", 'r')
    # file = open("./Data/Best/article/article_" + text_num_str + ".txt", 'r')
    file = open("./Data/Best/novel/novel_" + text_num_str + ".txt", 'r')
    Lines = file.readline()
    count = 0
    # Strips the newline character
    line_counter = 0

    while True:
        line = file.readline().strip()
        if not line:
            break
        # Removing the html tags and enter sign in a single line (or at the end of it)
        # print("line_counter = {}".format(line_counter))
        line = remove_tags2(line, "<NE>", "</NE>")
        line = remove_tags2(line, "<AB>", "</AB>")
        # Adding "|" to the end of each line if it is not there
        if len(line) >= 1 and line[len(line)-1] != '|':
            line += "|"
        # removable_strings = ["<NE>", "</NE>", "<AB>", "</AB>", "\n"]
        # removable_strings = ["\n"]
        # for bad_str in removable_strings:
        #     line = line.replace(bad_str, "")
        if "http" in line or len(line) == 0:
            continue
        # Adding "|" to the start of each line if it is not there
        if line[0] != '|':
            line = '|'+line
        # Finding word breakpoints
        word_brkpoints = []
        found_bars = 0
        for i in range(len(line)):
            if line[i] == '|':
                word_brkpoints.append(i-found_bars)
                found_bars += 1
        unsegmented_line = line.replace("|", "")

        # Detect different grapheme clusters in each line
        chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getUS())
        chars_break_iterator.setText(unsegmented_line)
        char_brkpoints = [0]
        for brkpoint in chars_break_iterator:
            char_brkpoints.append(brkpoint)

        for i in range(len(char_brkpoints) - 1):
            grapheme_clusters_dic[unsegmented_line[char_brkpoints[i]: char_brkpoints[i + 1]]] = grapheme_clusters_dic.get(
                unsegmented_line[char_brkpoints[i]: char_brkpoints[i + 1]], 0) + 1

        # Compute bies for the segmented data
        # print("****************************************************************")
        # print("line_counter = {}".format(line_counter))
        # print("LINE {} - TRUE SEG LINE   : {}".format(line_counter, line))
        # print("word_brkpoints = {}".format(word_brkpoints))
        # print("char_brkpoints = {}".format(char_brkpoints))

        true_bies = get_bies(char_brkpoints, word_brkpoints)
        true_bies_str = get_bies_string(true_bies)

        # Compute segmentation of icu and bies associated with it
        words_break_iterator = BreakIterator.createWordInstance(Locale.getUS())
        words_break_iterator.setText(unsegmented_line)
        icu_word_brkpoints = [0]
        for brkpoint in words_break_iterator:
            icu_word_brkpoints.append(brkpoint)
        icu_word_segmented_str = get_segmented_string(unsegmented_line, icu_word_brkpoints)
        icu_bies = get_bies(char_brkpoints, icu_word_brkpoints)
        icu_bies_str = get_bies_string(icu_bies)

        # Counting the number of mismatches between icu_bies and true_bies
        icu_total_bies_lengths += len(icu_bies_str)
        for i in range(len(icu_bies_str)):
            tru = true_bies_str[i]
            icu = icu_bies_str[i]
            if tru != icu:
                icu_mismatch += 1

        char_segmented_str = get_segmented_string(unsegmented_line, char_brkpoints)
        # print("LINE {} - UNSEG LINE      : {}".format(line_counter, unsegmented_line))
        # print("LINE {} - TRUE SEG LINE   : {}".format(line_counter, line))
        # print("LINE {} - ICU SEG LINE    : {}".format(line_counter, icu_word_segmented_str))
        # print("LINE {} - CHARACTERS      : {}".format(line_counter, char_segmented_str))
        # print("LINE {} - TRUE BIES STRING: {}".format(line_counter, true_bies_str))
        # print("LINE {} -  ICU BIES STRING: {}".format(line_counter, icu_bies_str))
        # print("LINE {} - TRUE WORD BREAKS: {}".format(line_counter, word_brkpoints))
        # print('**********************************************************************************')

        # print("LINE {}: {}".format(line_counter, word_brkpoints))

        line_counter += 1


graph_clust_freq = grapheme_clusters_dic
graph_clust_freq = {k: v for k, v in sorted(graph_clust_freq.items(), key=lambda item: item[1], reverse=True)}
graph_clust_ratio = graph_clust_freq
total = sum(graph_clust_ratio.values(), 0.0)
graph_clust_ratio = {k: v / total for k, v in graph_clust_ratio.items()}

# np.save(os.getcwd() + '/Data/graph_clust_ratio.npy', graph_clust_ratio)
'''
graph_clust_ratio = np.load(os.getcwd() + '/Data/graph_clust_ratio.npy', allow_pickle=True).item()
# print(graph_clust_ratio)
cnt = 0
thrsh = 350  # The vocabulary size for embeddings
thrsh = 700  # The vocabulary size for embeddings
grapheme_clusters_ids = dict()
for key in graph_clust_ratio.keys():
    if cnt < thrsh-1:
        grapheme_clusters_ids[key] = cnt
    if cnt == thrsh-1:
        break
    cnt += 1

# Assessing the precision of icu algorithm
# print("The bies precision of icu algorithm is {}".format(1 - icu_mismatch/icu_total_bies_lengths))

# Analyzing grapheme clusters for embeddings
'''
graph_clust_freq = grapheme_clusters_dic
graph_clust_freq = {k: v for k, v in sorted(graph_clust_freq.items(), key=lambda item: item[1], reverse=True)}
graph_clust_ratio = graph_clust_freq
total = sum(graph_clust_ratio.values(), 0.0)
graph_clust_ratio = {k: v / total for k, v in graph_clust_ratio.items()}
print(graph_clust_freq)
print("number of different grapheme clusters = {}".format(len(graph_clust_freq.keys())))
print(graph_clust_ratio)
thrsh = 0.9999
sum = 0
cnt = 0
for val in graph_clust_ratio.values():
    sum += val
    cnt += 1
    if sum > thrsh:
        break
print("{} grapheme clusters form {} of the text".format(cnt, thrsh))
plt.hist(graph_clust_freq.values(), bins=50)
plt.show()
# plt.savefig("./Figure/hist_graph_clust_freq.png")
'''

# line = "ผม|นอน|ดู|การ|ถ่ายทอด|ฟุตบอล|คู่|สำคัญ|ทาง|โทรทัศน์|ตั้งแต่|บ่าย| |แล้ว|งีบ|หลับ|ไป|เสีย|นาน| |ตื่น|ขึ้น|มา| |แสง|แดด|สลัว|ลง|ไป|มาก| |อีก|ไม่|ถึง|ชั่วโมง|ท้อง|ฟ้า|ก็|จะ|มืด|แล้ว| |ทั้ง|บ้าน|เงียบ|สงัด| |<NE>เสาวภาคย์</NE>|เมีย|ผม|คง|ยัง|ไม่|กลับ|จาก|งาน|วัน|เกิด|<NE>คุณวิทู</NE>|"
# line = "abc| |<NE>def</NE>| |sahand|<AB>sahand</AB>| |ghi|"
# line = remove_tags2(line, "<NE>", "</NE>")
# line = remove_tags2(line, "<AB>", "</AB>")
# print(line)
# x = input()

word_segmenter = WordSegmenter(input_n=100, input_t=200000, input_graph_clust_dic=grapheme_clusters_ids,
                               input_embedding_dim=40, input_hunits=64, input_dropout_rate=0.2, input_feature_dim=1,
                               input_output_dim=4, input_epochs=10, input_training_data="BEST",
                               input_evaluating_data="BEST")
word_segmenter.train_model()
word_segmenter.test_model()
word_segmenter.test_text(cat="news", text_num=1)
x = input()

# Building the LSTM model using the segmented data
# '''
num_texts = 5
train_texts_first = 1
valid_texts_first = 10
test_texts_first = 30
input_str = get_BEST_text(starting_text=train_texts_first, ending_text=train_texts_first+num_texts)
print(len(input_str))
times = 100000  # Number of characters that we cover
n = 50         # length of each batches
x_data, y_data = get_trainable_data(input_str, grapheme_clusters_ids)
x_data = x_data[:times]
y_data = y_data[:times, :]
train_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

print(x_data.shape)
print(y_data.shape)

input_str = get_BEST_text(starting_text=valid_texts_first, ending_text=valid_texts_first+num_texts)
print(len(input_str))
x_data, y_data = get_trainable_data(input_str, grapheme_clusters_ids)
x_data = x_data[:times]
y_data = y_data[:times, :]
valid_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

input_str = get_BEST_text(starting_text=test_texts_first, ending_text=test_texts_first+num_texts)
print(len(input_str))
x_data, y_data = get_trainable_data(input_str, grapheme_clusters_ids)
x_data = x_data[:times]
y_data = y_data[:times, :]
test_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

hunits = 20
vocab = 20
model = Sequential()
model.add(Embedding(thrsh, vocab, input_length=n))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(hunits, return_sequences=True), input_shape=(n, 1)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(4, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_generator.generate(), steps_per_epoch=train_generator.times//train_generator.batch_size,
                    epochs=10, validation_data=valid_generator.generate(), validation_steps=valid_generator.times//
                                                                                            valid_generator.batch_size)

# Code that computes output of tf.prediction manually
'''
all_test_input, all_actual_y = test_generator.generate_all_batches()
all_y_hat = model.predict(all_test_input)
for batch_id in range(all_test_input.shape[0]):
    test_input = all_test_input[batch_id, :]
    actual_y = all_actual_y[batch_id, :, :]
    y_hat = all_y_hat[batch_id, :]

    # Forward LSTM
    embedarr = model.weights[0]
    embedarr = embedarr.numpy()
    weightLSTM = model.weights[1: 4]
    c_fw = np.zeros([1, hunits])
    h_fw = np.zeros([1, hunits])
    all_h_fw = np.zeros([len(test_input), hunits])
    for i in range(len(test_input)):
        input_graph_id = int(test_input[i])
        x_t = embedarr[input_graph_id, :]
        x_t = x_t.reshape(1, x_t.shape[0])
        h_fw, c_fw = compute_hc(weightLSTM, x_t, h_fw, c_fw)
        all_h_fw[i, :] = h_fw

    # Backward LSTM
    embedarr = model.weights[0]
    embedarr = embedarr.numpy()
    weightLSTM = model.weights[4: 7]
    c_bw = np.zeros([1, hunits])
    h_bw = np.zeros([1, hunits])
    all_h_bw = np.zeros([len(test_input), hunits])
    for i in range(len(test_input)-1, -1, -1):
        input_graph_id = int(test_input[i])
        x_t = embedarr[input_graph_id, :]
        x_t = x_t.reshape(1, x_t.shape[0])
        h_bw, c_bw = compute_hc(weightLSTM, x_t, h_bw, c_bw)
        all_h_bw[i, :] = h_bw

    timew = model.weights[7]
    timew = timew.numpy()
    timeb = model.weights[8]
    timeb = timeb.numpy()
    est = np.zeros([len(test_input), 4])
    for i in range(len(test_input)):
        final_h = np.concatenate((all_h_fw[i, :], all_h_bw[i, :]), axis=0)
        final_h = final_h.reshape(1, 2*hunits)
        curr_est = final_h.dot(timew) + timeb
        curr_est = curr_est[0]
        curr_est = np.exp(curr_est)/sum(np.exp(curr_est))
        est[i, :] = curr_est

    for i in range(len(test_input)):
        print("Batch size = {}, Test case {}".format(batch_id, i))
        # print(y_hat[i, :])
        # print(est[i, :])
        print("tf bies    : {}".format(get_bies_string_from_softmax(y_hat)))
        print("manual bies: {}".format(get_bies_string_from_softmax(est)))
'''

# Testing the trained model using model.predict()
# '''
all_test_input, all_actual_y = test_generator.generate_all_batches()
all_y_hat = model.predict(all_test_input)
test_acc = []
for i in range(times//n):  # for each batch
    # print("test batch = {}".format(i))
    test_input = all_test_input[i, :]
    actual_y = all_actual_y[i, :, :]
    actual_y = get_bies_string(np.transpose(actual_y))
    print(actual_y)
    y_hat = all_y_hat[i, :, :]
    y_hat = get_bies_string_from_softmax(y_hat)
    print(y_hat)
    mismatch = 0
    for j in range(len(actual_y)):
        if actual_y[j] != y_hat[j]:
            mismatch += 1
    test_acc.append(1 - mismatch/len(actual_y))

test_acc = np.array(test_acc)
# print("test accuracy: \n{}".format(test_acc))
print("the average test accuracy: {}".format(np.mean(test_acc)))
# '''


# Checking LSTM for the case where the length of input text is not necessarily n
'''
all_test_input, all_actual_y = test_generator.generate_all_batches()
all_y_hat = model.predict(all_test_input)
test_acc1 = []
test_acc_sep = []
test_acc_comb = []
for i in range(times//n - 1):  # for each batch
    # print("test batch = {}".format(i))
    test_input1 = all_test_input[i, :]
    test_input2 = all_test_input[i+1, :]
    test_input = np.concatenate((test_input1, test_input2), axis=0)
    test_input = np.reshape(test_input, newshape=[1, test_input.shape[0]])
    y_hat_comb = model.predict(test_input)
    y_hat_comb = get_bies_string_from_softmax(y_hat_comb[0, :, :])
    # print("combined y_hat = {}".format(y_hat_comb))

    y_hat1 = all_y_hat[i, :, :]
    y_hat1 = get_bies_string_from_softmax(y_hat1)
    y_hat2 = all_y_hat[i+1, :, :]
    y_hat2 = get_bies_string_from_softmax(y_hat2)
    y_hat_sep = y_hat1 + y_hat2
    # print("separate y_hat = {}".format(y_hat_sep))

    actual_y1 = all_actual_y[i, :, :]
    actual_y1 = get_bies_string(np.transpose(actual_y1))
    actual_y2 = all_actual_y[i+1, :, :]
    actual_y2 = get_bies_string(np.transpose(actual_y2))
    actual_y = actual_y1 + actual_y2
    # print("actual y       = {}".format(actual_y))

    # print(y_hat1.shape)
    # print(y_hat2.shape)
    # print(y_hat.shape)

    mismatch_sep = 0
    mismatch_comb = 0
    mismatch1 = 0
    for j in range(len(actual_y1)):
        if actual_y1[j] != y_hat1[j]:
            mismatch1 += 1
    test_acc1.append(1 - mismatch1 / len(actual_y1))
    # print(len(actual_y))
    # print(len(y_hat_sep))
    # print(len(y_hat_comb))
    # x = input()
    for j in range(len(actual_y)):
        if actual_y[j] != y_hat_sep[j]:
            mismatch_sep += 1
        if actual_y[j] != y_hat_comb[j]:
            mismatch_comb += 1
    test_acc_sep.append(1 - mismatch_sep/len(actual_y))
    test_acc_comb.append(1 - mismatch_comb/len(actual_y))

test_acc_acc1 = np.array(test_acc1)
test_acc_sep = np.array(test_acc_sep)
test_acc_comb = np.array(test_acc_comb)
print("the average test accuracy for case 1   : {}".format(np.mean(test_acc1)))
print("the average test accuracy for seperated: {}".format(np.mean(test_acc_sep)))
print("the average test accuracy for combined : {}".format(np.mean(test_acc_comb)))
plt.plot(test_acc_sep, label="seperate")
plt.plot(test_acc_comb, label="combined")
plt.legend()
plt.show()
# print("test accuracy: \n{}".format(test_acc))

'''


# Building the LSTM model using the segmented data
'''
num_texts = 5
train_texts_first = 1
valid_texts_first = 10
test_texts_first = 30
input_str = get_BEST_text(starting_text=train_texts_first, ending_text=train_texts_first+num_texts)
print(len(input_str))
times = 100000  # Number of characters that we cover
n = 50         # length of each batches
x_data, y_data = get_pseudo_trainable_data(input_str, times, n, grapheme_clusters_ids)
train_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

input_str = get_BEST_text(starting_text=valid_texts_first, ending_text=valid_texts_first+num_texts)
print(len(input_str))
x_data, y_data = get_pseudo_trainable_data(input_str, times, n, grapheme_clusters_ids)
valid_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

input_str = get_BEST_text(starting_text=test_texts_first, ending_text=test_texts_first+num_texts)
print(len(input_str))
x_data, y_data = get_trainable_data(input_str, times, n, grapheme_clusters_ids)
test_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

model = Sequential()
model.add(Embedding(thrsh, 20, input_length=n))
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n, 1)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(4, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_generator.generate(), steps_per_epoch=train_generator.times//train_generator.batch_size,
                    epochs=10, validation_data=valid_generator.generate(), validation_steps=valid_generator.times//
                                                                                            valid_generator.batch_size)
all_test_input, all_actual_y = test_generator.generate_all_batches()
all_y_hat = model.predict(all_test_input)
test_acc = []
for i in range(times//n):  # for each batch
    print("test batch = {}".format(i))
    test_input = all_test_input[i, :]
    actual_y = all_actual_y[i, :, :]
    actual_y = get_bies_string(np.transpose(actual_y))
    print(actual_y)
    y_hat = all_y_hat[i, :, :]
    y_hat = get_bies_string_from_softmax(y_hat)
    print(y_hat)
    mismatch = 0
    for i in range(len(actual_y)):
        if actual_y[i] != y_hat[i]:
            mismatch += 1
    test_acc.append(1 - mismatch/len(actual_y))

test_acc = np.array(test_acc)
print("test accuracy: \n{}".format(test_acc))
print("the average test accuracy: {}".format(np.mean(test_acc)))

'''






