import numpy as np
import os
import icu
import datetime
from icu import UnicodeString, BreakIterator, Locale
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Dropout
# from keras import optimizer


def is_all_english(str):
    """
    A very basic function that checks if all elements of str are english characters or common expressions
    It just checks if the integer assigned to each character is below 130 or not
    TO DO: This function can be potentially be replace with some ICU function
    str: input string
    """

    str = str.replace(" ", "")
    str = str.replace("|", "")
    # for k in range(0, 130):
    #     print(chr(k))
    for ch in str:
        if ord(ch) > 130:
            # print("character = {}, ord = {}".format(ch, ord(ch)))
            return False
    return True


def sigmoid(x):
    """
    Computes the sigmoid function for a scalar
    x: the scalar
    """
    return 1.0/(1.0+np.exp(-x))


def analyze_grapheme_clusters(ratios, thrsh):
    """
    This function analyzes the grapheme clusters, to see what percentage of them form which percent of the text, and
    provides a histogram that shows frequency of grapheme clusters
    ratios: a dictionary that holds the ratio of text that is represented by each grapheme cluster
    thrsh: shows what percent of the text we want to cover
    """
    cum_sum = 0
    cnt = 0
    for val in ratios.values():
        cum_sum += val
        cnt += 1
        if cum_sum > thrsh:
            break
    print("number of different grapheme clusters = {}".format(len(ratios.keys())))
    print("{} grapheme clusters form {} of the text".format(cnt, thrsh))
    # plt.hist(ratios.values(), bins=50)
    # plt.show()


def get_segmented_string(str, brkpoints):
    """
    Rerurns a segmented string using the unsegmented string and and break points. Simply inserts | at break points.
    str: unsegmented string
    brkpoints: break points
    """
    out = "|"
    for i in range(len(brkpoints)-1):
        out += str[brkpoints[i]: brkpoints[i+1]] + "|"
    return out


def get_bies(char_brkpoints, word_brkpoints):
    """
    Given break points for words and grapheme clusters, returns the matrix that represents BIES.
    The output is a matrix of size (n * 4) where n is the number of grapheme clusters in the string
    char_brkpoints: break points for grapheme clusters
    word_brkpoints: break points for words
    """
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


def get_bies_string_from_softmax (mat):
    """
    Computes estimated BIES based on a softmax matrix of size n*4.
    Each row of matrix gives four floats that add up to 1, which are probability of B, I, E,  and S.
    This function simply pick the one with highest probability. In ties, it picks B over I, I over E, and E over S.
    mat: Input matrix that contains softmax probabilities. Dimension is n*4 where n is length of the string.
    """
    out = ""
    for i in range(mat.shape[0]):
        max_softmax = max(mat[i, :])
        if mat[i, 0] == max_softmax:
            out += "b"
        elif mat[i, 1] == max_softmax:
            out += "i"
        elif mat[i, 2] == max_softmax:
            out += "e"
        elif mat[i, 3] == max_softmax:
            out += "s"
    return out


def remove_tags(line, st_tag, fn_tag):
    """
    Given a string and two substrings, remove any text between these tags.
    line: the input string
    st_tag: the first substring
    fn_tag: the secibd substring
    It handles spaces around tags as follows:
        abc|<NE>def</NE>|ghi      ---> abc|ghi
        abc| |<NE>def</NE>|ghi    ---> abc| |ghi
        abc|<NE>def</NE>| |ghi    ---> abc| |ghi
        abc| |<NE>def</NE>| |ghi  ---> abc| |ghi
    """

    new_line = ""
    st_ind = 0
    while st_ind < len(line):
        if line[st_ind: st_ind+len(st_tag)] == st_tag:
            fn_ind = st_ind
            while fn_ind < len(line):
                if line[fn_ind: fn_ind+len(fn_tag)] == fn_tag:
                    fn_ind = fn_ind+len(fn_tag) + 1
                    if st_ind -2 >= 0 and fn_ind+2 <= len(line):
                        if line[st_ind-2:st_ind] == " |" and line[fn_ind:fn_ind+2] == " |":
                            fn_ind += 2
                    st_ind = fn_ind
                    break
                else:
                    fn_ind += 1
        if st_ind < len(line):
            new_line += line[st_ind]
        st_ind += 1
    return new_line


def clean_line(line):
    """
    This line cleans a line as follows such that it is ready for process by different components of the code. It returns
    the clean line or -1, if the line should be omitted.
        1) remove tags and https from the line.
        2) Put a | at the begining and end of the line if it isn't already there
        3) if line is very short (len < 3) or if it is all in English or it has a link in it, return -1
    """

    # Remove lines with links
    if "http" in line or len(line) == 0:
        return -1

    # Remove texts between following tags
    line = remove_tags(line, "<NE>", "</NE>")
    line = remove_tags(line, "<AB>", "</AB>")

    # Remove lines that are all fully in English
    if is_all_english(line):
        return -1

    # Exclude cases such as "", " ", "| ", " |", etc.
    if len(line) < 3:
        return -1

    # Add "|" to the end of each line if it is not there
    if len(line) >= 1 and line[len(line) - 1] != '|':
        line += "|"

    # Adding "|" to the start of each line if it is not there
    if line[0] != '|':
        line = '|' + line

    return line


def preprocess():
    """
    This function uses the BEST data set to
        1) compute the grapheme cluster dictionary that holds the frequency of different grapheme clusters
        2) demonstrate the performance of icu word breakIterator and compute its accuracy
    """
    grapheme_clusters_dic = dict()
    icu_mismatch = 0
    icu_total_bies_lengths = 0
    for cat in ["news", "encyclopedia", "article", "novel"]:
        for text_num in range(1, 96):
            text_num_str = "{}".format(text_num).zfill(5)
            file = open("./Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt", 'r')
            line_counter = 0
            while True:
                line = file.readline().strip()
                if not line:
                    break

                line = clean_line(line)
                if line == -1:
                    continue

                # Finding word breakpoints using the segmented data
                word_brkpoints = []
                found_bars = 0
                for i in range(len(line)):
                    if line[i] == '|':
                        word_brkpoints.append(i - found_bars)
                        found_bars += 1

                # Creating the unsegmented line
                unsegmented_line = line.replace("|", "")

                # Making the grapheme clusters brkpoints
                chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getUS())
                chars_break_iterator.setText(unsegmented_line)
                char_brkpoints = [0]
                for brkpoint in chars_break_iterator:
                    char_brkpoints.append(brkpoint)

                # Storing the grapheme clusters and their frequency in the dictionary
                for i in range(len(char_brkpoints) - 1):
                    grapheme_clusters_dic[
                        unsegmented_line[char_brkpoints[i]: char_brkpoints[i + 1]]] = grapheme_clusters_dic.get(
                        unsegmented_line[char_brkpoints[i]: char_brkpoints[i + 1]], 0) + 1
                true_bies = get_bies(char_brkpoints, word_brkpoints)
                true_bies_str = get_bies_string_from_softmax(np.transpose(true_bies))

                # Compute segmentations of icu and BIES associated with it
                words_break_iterator = BreakIterator.createWordInstance(Locale.getUS())
                words_break_iterator.setText(unsegmented_line)
                icu_word_brkpoints = [0]
                for brkpoint in words_break_iterator:
                    icu_word_brkpoints.append(brkpoint)
                icu_word_segmented_str = get_segmented_string(unsegmented_line, icu_word_brkpoints)
                icu_bies = get_bies(char_brkpoints, icu_word_brkpoints)
                icu_bies_str = get_bies_string_from_softmax(np.transpose(icu_bies))

                # Counting the number of mismatches between icu_bies and true_bies
                icu_total_bies_lengths += len(icu_bies_str)
                for i in range(len(icu_bies_str)):
                    tru = true_bies_str[i]
                    icu = icu_bies_str[i]
                    if tru != icu:
                        icu_mismatch += 1

                # Demonstrate how icu segmenter works
                # char_segmented_str = get_segmented_string(unsegmented_line, char_brkpoints)
                # print("Cat: {}, Text number: {}".format(cat, text_num))
                # print("LINE {} - UNSEG LINE      : {}".format(line_counter, unsegmented_line))
                # print("LINE {} - TRUE SEG LINE   : {}".format(line_counter, line))
                # print("LINE {} - ICU SEG LINE    : {}".format(line_counter, icu_word_segmented_str))
                # print("LINE {} - CHARACTERS      : {}".format(line_counter, char_segmented_str))
                # print("LINE {} - TRUE BIES STRING: {}".format(line_counter, true_bies_str))
                # print("LINE {} -  ICU BIES STRING: {}".format(line_counter, icu_bies_str))
                # print("LINE {} - TRUE WORD BREAKS: {}".format(line_counter, word_brkpoints))
                # print('**********************************************************************************')

                line_counter += 1

    icu_accuracy = 1 - icu_mismatch/icu_total_bies_lengths
    graph_clust_freq = grapheme_clusters_dic
    graph_clust_freq = {k: v for k, v in sorted(graph_clust_freq.items(), key=lambda item: item[1], reverse=True)}
    graph_clust_ratio = graph_clust_freq
    total = sum(graph_clust_ratio.values(), 0.0)
    graph_clust_ratio = {k: v / total for k, v in graph_clust_ratio.items()}

    return graph_clust_ratio, icu_accuracy


def add_space_bars(read_filename, write_filename):
    rfile = open(read_filename, 'r')
    wfile = open(write_filename, 'w')
    while True:
        line = rfile.readline().strip()
        if not line:
            break
        new_line = ""
        for ch in line:
            if ch == " ":
                new_line += "| |"
            else:
                new_line += ch
        new_line += "\n"
        wfile.write(new_line)


def compute_ICU_accuracy(filename):
    """
    This function uses a dataset to compute the accuracy of icu word breakIterator
    filename: The path of the file
    """
    file = open(filename, 'r')
    line_counter = 0
    icu_mismatch = 0
    icu_total_bies_lengths = 0
    while True:
        line = file.readline().strip()
        if not line:
            break

        line = clean_line(line)
        if line == -1:
            continue

        # Finding word breakpoints using the segmented data
        word_brkpoints = []
        found_bars = 0
        for i in range(len(line)):
            if line[i] == '|':
                word_brkpoints.append(i - found_bars)
                found_bars += 1

        # Creating the unsegmented line
        unsegmented_line = line.replace("|", "")

        # # Making the grapheme clusters brkpoints
        chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getUS())
        chars_break_iterator.setText(unsegmented_line)
        char_brkpoints = [0]
        for brkpoint in chars_break_iterator:
            char_brkpoints.append(brkpoint)

        true_bies = get_bies(char_brkpoints, word_brkpoints)
        true_bies_str = get_bies_string_from_softmax(np.transpose(true_bies))

        # Compute segmentations of icu and BIES associated with it
        words_break_iterator = BreakIterator.createWordInstance(Locale.getUS())
        words_break_iterator.setText(unsegmented_line)
        icu_word_brkpoints = [0]
        for brkpoint in words_break_iterator:
            icu_word_brkpoints.append(brkpoint)

        icu_word_segmented_str = get_segmented_string(unsegmented_line, icu_word_brkpoints)
        icu_bies = get_bies(char_brkpoints, icu_word_brkpoints)
        icu_bies_str = get_bies_string_from_softmax(np.transpose(icu_bies))

        # Counting the number of mismatches between icu_bies and true_bies
        icu_total_bies_lengths += len(icu_bies_str)
        for i in range(len(icu_bies_str)):
            tru = true_bies_str[i]
            icu = icu_bies_str[i]
            if tru != icu:
                icu_mismatch += 1

        # Demonstrate how icu segmenter works
        # char_segmented_str = get_segmented_string(unsegmented_line, char_brkpoints)
        # print("Cat: {}, Text number: {}".format(cat, text_num))
        # print("LINE {} - UNSEG LINE      : {}".format(line_counter, unsegmented_line))
        # print("LINE {} - TRUE SEG LINE   : {}".format(line_counter, line))
        # print("LINE {} - ICU SEG LINE    : {}".format(line_counter, icu_word_segmented_str))
        # print("LINE {} - CHARACTERS      : {}".format(line_counter, char_segmented_str))
        # print("LINE {} - TRUE BIES STRING: {}".format(line_counter, true_bies_str))
        # print("LINE {} -  ICU BIES STRING: {}".format(line_counter, icu_bies_str))
        # print("LINE {} - TRUE WORD BREAKS: {}".format(line_counter, word_brkpoints))
        # print('**********************************************************************************')

        line_counter += 1
    icu_accuracy = 1 - icu_mismatch / icu_total_bies_lengths
    return icu_accuracy


def get_BEST_text(starting_text, ending_text):
    """
    Gives a long string, that contains all lines (separated by a single space) from BEST data with numbers in a range
    This function use data from all sources (news, encyclopedia, article, and novel)
    It removes all texts between pair of tags (<NE>, </NE>) and (<AB>, </AB>), assures that the string ends with a "|",
    and ignores empty lines, lines with "http" in them, and lines that are all in english (since these are not segmented
    in the BEST data set)
    starting_text: number or the smallest text
    ending_text: number or the largest text + 1
    """
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
                line = clean_line(line)
                if line == -1:
                    continue
                if len(out_str) == 0:
                    out_str = line
                else:
                    out_str = out_str + " " + line
                line_counter += 1
    return out_str


def get_file_text(filename):
    """
    Gives a long string, that contains all lines (separated by a single space) from a file as well as a list of points
    to starts and ends of actual lines. Each two consecutive lines are separated by a space.
    It removes all texts between pair of tags (<NE>, </NE>) and (<AB>, </AB>), assures that each line starts and ends
    with a "|", and ignores empty lines, lines with "http" in them, and lines that are all in english
    (since these are usually not segmented)
    """
    file = open(filename, 'r')
    line_counter = 0
    out_str = ""
    while True:
        line = file.readline().strip()
        if not line:
            break
        line = clean_line(line)
        if line == -1:
            continue
        if len(out_str) == 0:
            out_str = line
        else:
            out_str = out_str + " " + line
        line_counter += 1
    return out_str


def get_trainable_data(input_line, graph_clust_ids):
    """
    Given a segmented line, extracts x_data (with respect to a dictionary that maps grapheme clusters to integers)
    and y_data which is a n*4 matrix that represents BIES where n is the length of the unsegmented line. All grapheme
    clusters not found in the dictionary are set to the largest value of the dictionary plus 1
    input_line: the unsegmented line
    graph_clust_ids: a dictionary that stores maps from grapheme clusters to integers
    """
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

    # Finding BIES
    true_bies = get_bies(char_brkpoints, word_brkpoints)
    true_bies_str = get_bies_string_from_softmax(np.transpose(true_bies))

    # Making x_data and y_data
    times = len(char_brkpoints)-1
    x_data = np.zeros(shape=[times, 1])
    y_data = np.zeros(shape=[times, 4])
    excess_grapheme_ids = max(graph_clust_ids.values()) + 1
    for i in range(times):
        char_st = char_brkpoints[i]
        char_fn = char_brkpoints[i + 1]
        curr_char = unsegmented_line[char_st: char_fn]
        x_data[i, 0] = graph_clust_ids.get(curr_char, excess_grapheme_ids)
        y_data[i, :] = true_bies[:, i]
    return x_data, y_data


def compute_hc(weight, x_t, h_tm1, c_tm1):
    """
    Given weights of a LSTM model, the input at time t, and values for h and c at time t-1, compute the values of h and
    c for time t.
    weights: a list of three matrices, which are W (from input to cell), U (from h to cell), and b (bias) respectively.
    Dimensions: warr.shape = (nfeature, hunits*4), uarr.shape = (hunits, hunits*4), barr.shape = (hunits*4,)
    """
    warr, uarr, barr = weight
    warr = warr.numpy()
    uarr = uarr.numpy()
    barr = barr.numpy()

    # Implementing gates (forget, input, and output)
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
    """
    A batch generator component, which is used to generate batches for training, validation, and evaluation. The current
    version works only for inputs of dimension 1.
    x_data: The input of the model.
    y_data: The output of the model
    time_steps: length of the input and output in each batch
    batch_size: number of batches
    dim_output: dimension of the output
    """
    def __init__(self, x_data, y_data, n, batch_size, dim_output):
        self.x_data = x_data  # dim = times * dim_features
        self.y_data = y_data  # dim = times * dim_output
        self.n = n
        self.batch_size = batch_size
        self.dim_output = dim_output
        if x_data.shape[0] < batch_size * n or y_data.shape[0] < batch_size * n:
            print("Warning: x_data or y_data is not large enough!")

    def generate(self):
        """
        generates batches one by one, used for training and validation
        """
        x = np.zeros([self.batch_size, self.n])
        y = np.zeros([self.batch_size, self.n, self.dim_output])
        while True:
            for i in range(self.batch_size):
                x[i, :] = self.x_data[self.n * i: self.n * (i + 1), 0]
                y[i, :, :] = self.y_data[self.n * i: self.n * (i + 1), :]
            yield x, y

    def generate_all_batches(self):
        """
        returns all batches together, used mostly for testing
        """
        x = np.zeros([self.batch_size, self.n])
        y = np.zeros([self.batch_size, self.n, self.dim_output])
        for i in range(self.batch_size):
            x[i, :] = self.x_data[self.n * i: self.n * (i + 1), 0]
            y[i, :, :] = self.y_data[self.n * i: self.n * (i + 1), :]
        return x, y


class WordSegmenter:
    """
    A class that let you make a bi-directional LSTM, train it, and test it. It assumes that the number of features is 1.
    n: Length of the input for LSTM model
    t: The total length of data used to train and validate the model. It is equal to number of batches times n
    batch_size: number of batches used to train the model. It is defined as t // n
    graph_clust_dic: a dictionary that maps the most frequent grapheme clusters to integers
    clusters_num: number of grapheme clusters in graph_clust_dic
    embedding_dim: length of the embedding vectors for each grapheme cluster
    hunits: number of units used in each cell of LSTM
    dropout_rate: dropout rate used in layers after the embedding and after the bidirectional LSTM
    output_dim: dimension of the output layer
    epochs: number of epochs used to train the model
    training_data: name of the data used to train the model
    evaluating_data: name of the data used to evaluate the model
    """
    def __init__(self, input_n, input_t, input_graph_clust_dic, input_embedding_dim, input_hunits, input_dropout_rate,
                 input_output_dim, input_epochs, input_training_data, input_evaluating_data):
        self.n = input_n
        self.t = input_t
        if self.t % self.n != 0:
            print("Warning: t is not divided by n")
        self.batch_size = self.t // self.n
        self.graph_clust_dic = input_graph_clust_dic
        self.clusters_num = len(self.graph_clust_dic.keys()) + 1
        self.embedding_dim = input_embedding_dim
        self.hunits = input_hunits
        self.dropout_rate = input_dropout_rate
        self.output_dim = input_output_dim
        self.epochs = input_epochs
        self.training_data = input_training_data
        self.evaluating_data = input_evaluating_data
        self.model = None

    def train_model(self):
        """
        This function trains the model using the dataset specified in the __init__ function. It combine all sentences in
        the data set with a space between them and then divide this large string into strings of fixed length self.n.
        Therefore, it may (and probably will) break some words and puts different part of them in different batches.
        """

        # Get training data of length self.t
        x_data = []
        y_data = []
        if self.training_data == "BEST":
            # this chunk of data has ~ 2*10^6 data points
            input_str = get_BEST_text(starting_text=1, ending_text=10)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            if self.t > x_data.shape[0]:
                print("Warning: size of the training data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        else:
            print("Warning: no implementation for this training data exists!")
        train_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=self.batch_size,
                                              dim_output=self.output_dim)

        # Get validation data of length self.t
        if self.training_data == "BEST":
            # this chunk of data has ~ 2*10^6 data points
            input_str = get_BEST_text(starting_text=10, ending_text=20)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            if self.t > x_data.shape[0]:
                print("Warning: size of the validation data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        else:
            print("Warning: no implementation for this validation data exists!")
        valid_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=self.batch_size,
                                              dim_output=self.output_dim)

        # Building the model
        model = Sequential()
        model.add(Embedding(self.clusters_num, self.embedding_dim, input_length=self.n))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(self.hunits, return_sequences=True), input_shape=(self.n, 1)))
        model.add(Dropout(self.dropout_rate))
        model.add(TimeDistributed(Dense(self.output_dim, activation='softmax')))
        opt = keras.optimizers.Adam(learning_rate=0.1)
        # opt = keras.optimizers.SGD(learning_rate=0.4, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

        # Fitting the model
        model.fit(train_generator.generate(), steps_per_epoch=self.t//self.batch_size,
                  epochs=self.epochs, validation_data=valid_generator.generate(),
                  validation_steps=self.t//self.batch_size)
        self.model = model

    def test_model(self):
        """
        This function tests the model fitted in self.train(). It uses the same format (combining all sentences separated
         by spaces) to test the model.
        """
        # Get test data
        x_data = []
        y_data = []
        if self.evaluating_data == "BEST":
            # this chunk of data has ~ 2*10^6 data points
            input_str = get_BEST_text(starting_text=40, ending_text=45)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
        elif self.evaluating_data == "SAFT":
            input_str = get_file_text("./Data/SAFT/test.txt")
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
        else:
            print("Warning: no implementation for this evaluation data exists!")
        test_batch_size = x_data.shape[0]//self.n
        test_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=test_batch_size,
                                             dim_output=self.output_dim)

        # Testing batch by batch (each batch of length self.n)
        all_test_input, all_actual_y = test_generator.generate_all_batches()
        all_y_hat = self.model.predict(all_test_input)
        test_acc = []
        for i in range(test_batch_size):
            actual_y = all_actual_y[i, :, :]
            actual_y = get_bies_string_from_softmax(actual_y)
            y_hat = all_y_hat[i, :, :]
            y_hat = get_bies_string_from_softmax(y_hat)

            # Compute the BIES accuracy
            mismatch = 0
            for j in range(len(actual_y)):
                if actual_y[j] != y_hat[j]:
                    mismatch += 1
            test_acc.append(1 - mismatch / len(actual_y))
        test_acc = np.array(test_acc)
        print("the average test accuracy in test_model function: {}".format(np.mean(test_acc)))
        return np.mean(test_acc)

    def test_text_line_by_line(self, cat, text_num):
        """
        This function tests the model fitted in self.train() using BEST data set. Unlike test_model() function, this
        function tests the model line by line. It combines very short lines together before testing.
        cat: category of the text in BEST data. It can be "news", "encyclopedia", "article" or "novel"
        text_num: number of the text in the BEST data
        """
        text_num_str = "{}".format(text_num).zfill(5)
        file = open("./Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt", 'r')
        test_acc = []
        prev_str = ""
        while True:
            line = file.readline().strip()
            if not line:
                break
            line = clean_line(line)
            if line == -1:
                continue

            # If the new line is too short, combine it with previous short lines. Process it if it gets long enough.
            # If this value is set to infinity, basically we are converting the whole text into one big string and
            # evaluating that; just like test_model() function
            if len(line) < 30:
                prev_str = prev_str + line
                if len(prev_str) >= 50:
                    line = prev_str
                    prev_str = ""
                else:
                    continue

            # Get trainable data
            x_data, y_data = get_trainable_data(line, self.graph_clust_dic)

            # Use the manual predict function -- the tf function doesn't always work properly for varying length strings
            y_hat = self.manual_predict(x_data)
            y_hat = get_bies_string_from_softmax(y_hat)
            actual_y = get_bies_string_from_softmax(y_data)

            # Compute the BIES accuracy
            mismatch = 0
            for j in range(len(actual_y)):
                if actual_y[j] != y_hat[j]:
                    mismatch += 1
            test_acc.append(1 - mismatch / len(actual_y))
        print("the average test accuracy (line by line) for text {} : {}".format(text_num, np.mean(test_acc)))
        return test_acc

    def test_model_line_by_line(self):
        """
        This function uses the test_text_line_by_line() to test the model by a range of texts in BEST data set. The
        final score is the average of scores computed for each individual text.
        """
        all_test_acc = []
        category = ["news", "encyclopedia", "article", "novel"]
        for text_num in range(30, 33):
            print("testing text {}".format(text_num))
            for cat in category:
                all_test_acc += self.test_text_line_by_line(cat, text_num)
        print("the average test accuracy by test_model_line_by_line function: {}".format(np.mean(all_test_acc)))
        return np.mean(all_test_acc)

    def manual_predict(self, test_input):
        """
        Implementation of the tf.predict function manually. This function works for inputs of any length, and only uses
        model weights obtained from self.model.weights.
        test_input: the input text
        """
        # Forward LSTM
        embedarr = self.model.weights[0]
        embedarr = embedarr.numpy()
        weightLSTM = self.model.weights[1: 4]
        c_fw = np.zeros([1, self.hunits])
        h_fw = np.zeros([1, self.hunits])
        all_h_fw = np.zeros([len(test_input), self.hunits])
        for i in range(len(test_input)):
            input_graph_id = int(test_input[i])
            x_t = embedarr[input_graph_id, :]
            x_t = x_t.reshape(1, x_t.shape[0])
            h_fw, c_fw = compute_hc(weightLSTM, x_t, h_fw, c_fw)
            all_h_fw[i, :] = h_fw

        # Backward LSTM
        embedarr = self.model.weights[0]
        embedarr = embedarr.numpy()
        weightLSTM = self.model.weights[4: 7]
        c_bw = np.zeros([1, self.hunits])
        h_bw = np.zeros([1, self.hunits])
        all_h_bw = np.zeros([len(test_input), self.hunits])
        for i in range(len(test_input) - 1, -1, -1):
            input_graph_id = int(test_input[i])
            x_t = embedarr[input_graph_id, :]
            x_t = x_t.reshape(1, x_t.shape[0])
            h_bw, c_bw = compute_hc(weightLSTM, x_t, h_bw, c_bw)
            all_h_bw[i, :] = h_bw

        # Combining Forward and Backward layers through dense time-distributed layer
        timew = self.model.weights[7]
        timew = timew.numpy()
        timeb = self.model.weights[8]
        timeb = timeb.numpy()
        est = np.zeros([len(test_input), 4])
        for i in range(len(test_input)):
            final_h = np.concatenate((all_h_fw[i, :], all_h_bw[i, :]), axis=0)
            final_h = final_h.reshape(1, 2 * self.hunits)
            curr_est = final_h.dot(timew) + timeb
            curr_est = curr_est[0]
            curr_est = np.exp(curr_est) / sum(np.exp(curr_est))
            est[i, :] = curr_est
        return est


# graph_clust_ratio, icu_accuracy = preprocess()
# print("icu accuracy is {}".format(icu_accuracy))
# np.save(os.getcwd() + '/Data/graph_clust_ratio.npy', graph_clust_ratio)
# analyze_grapheme_clusters(ratios=graph_clust_ratio, thrsh=0.999)

# Loading the graph_clust from memory
graph_clust_ratio = np.load(os.getcwd() + '/Data/graph_clust_ratio.npy', allow_pickle=True).item()

# Looking at the accuracy of the ICU on SAFT data set
# print("Accuracy of ICU on SAFT data is {}.".format(compute_ICU_accuracy(os.getcwd() + "/Data/SAFT/test.txt")))

# Making the grapheme cluster dictionary to be used in the bi-directional LSTM model
cnt = 0
graph_thrsh = 500  # The vocabulary size for embeddings
graph_clust_dic = dict()
for key in graph_clust_ratio.keys():
    if cnt < graph_thrsh-1:
        graph_clust_dic[key] = cnt
    if cnt == graph_thrsh-1:
        break
    cnt += 1

print("starting")
add_space_bars("./Data/SAFT/test_raw.txt", "./Data/SAFT/test.txt")
print("ending")


print("The accuracy of ICU for SAFT data is {}".format(compute_ICU_accuracy("./Data/SAFT/test.txt")))

x = input()

# Making the bi-directional LSTM model using BEST data set
word_segmenter = WordSegmenter(input_n=50, input_t=100000, input_graph_clust_dic=graph_clust_dic,
                               input_embedding_dim=40, input_hunits=40, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=10, input_training_data="BEST", input_evaluating_data="BEST")
word_segmenter.train_model()
word_segmenter.test_model()
# word_segmenter.test_model_line_by_line()


# Grid search
'''
test1 = []
hu_list = [10, 20, 40, 64, 128, 256]
for hu in hu_list:
    word_segmenter = WordSegmenter(input_n=50, input_t=100000, input_graph_clust_dic=graph_clust_dic,
                                   input_embedding_dim=40, input_hunits=hu, input_dropout_rate=0.2, input_output_dim=4
                                   , input_epochs=20, input_training_data="BEST", input_evaluating_data="BEST")
    word_segmenter.train_model()
    test1.append(word_segmenter.test_model())
print(test1)
'''
