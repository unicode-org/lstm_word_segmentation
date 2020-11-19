import os
import numpy as np
import json
from icu import Char
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Embedding, Dropout, Flatten
from tensorflow import keras
import tensorflow as tf

import constants
from helpers import sigmoid
from text_helpers import clean_line, combine_lines_of_file, get_best_data_text
from accuracy import Accuracy
from line import Line
from bies import Bies
from grapheme_cluster import GraphemeCluster


class KerasBatchGenerator(object):
    """
    A batch generator component, which is used to generate batches for training, validation, and evaluation. The current
    version works only for inputs of dimension 1.
    Args:
        x_data: The input of the model
        y_data: The output of the model
        n: length of the input and output in each batch
        batch_size: number of batches
        dim_output: dimension of the output
    """
    def __init__(self, x_data, y_data, n, batch_size, dim_output):
        self.x_data = x_data  # A list of GraphemeCluster objects
        self.y_data = y_data  # dim = times * dim_output
        self.n = n
        self.batch_size = batch_size
        self.dim_output = dim_output
        if len(x_data) < batch_size * n or y_data.shape[0] < batch_size * n:
            print("Warning: x_data or y_data is not large enough!")

    # Generating functions for grapheme clusters when the embedding layer is implemented by tf
    def generate(self, type):
        """
        generates batches one by one, used for training and validation when the tf embedding layer is used
        """
        y = np.zeros([self.batch_size, self.n, self.dim_output])
        if type == "grapheme_clusters_tf":
            x = np.zeros([self.batch_size, self.n])
        if type == "grapheme_clusters_man":
            x = np.zeros([self.batch_size, self.n, self.x_data[0].num_clusters])
        if type == "generalized_vectors":
            x = np.zeros([self.batch_size, self.n, self.x_data[0].generalized_vec_length])
        while True:
            for i in range(self.batch_size):
                for j in range(self.n):
                    if type == "grapheme_clusters_tf":
                        x[i, j] = self.x_data[self.n*i + j].graph_clust_id
                    if type == "grapheme_clusters_man":
                        x[i, j, :] = self.x_data[self.n*i + j].graph_clust_vec
                    if type == "generalized_vectors":
                        x[i, j, :] = self.x_data[self.n*i + j].generalized_vec
                y[i, :, :] = self.y_data[self.n * i: self.n * (i + 1), :]
            yield x, y

    def generate_all(self, type):
        """
        generates batches one by one, used for training and validation when the tf embedding layer is used
        """
        y = np.zeros([self.batch_size, self.n, self.dim_output])
        if type == "grapheme_clusters_tf":
            x = np.zeros([self.batch_size, self.n])
        if type == "grapheme_clusters_man":
            x = np.zeros([self.batch_size, self.n, self.x_data[0].num_clusters])
        if type == "generalized_vectors":
            x = np.zeros([self.batch_size, self.n, self.x_data[0].generalized_vec_length])

        for i in range(self.batch_size):
            for j in range(self.n):
                if type == "grapheme_clusters_tf":
                    x[i, j] = self.x_data[self.n*i + j].graph_clust_id
                if type == "grapheme_clusters_man":
                    x[i, j, :] = self.x_data[self.n*i + j].graph_clust_vec
                if type == "generalized_vectors":
                    x[i, j, :] = self.x_data[self.n*i + j].generalized_vec
            y[i, :, :] = self.y_data[self.n * i: self.n * (i + 1), :]
        return x, y


class WordSegmenter:
    """
    A class that let you make a bi-directional LSTM, train it, and test it. It assumes that the number of features is 1.
    Args:
        input_n: Length of the input for LSTM model
        input_t: The total length of data used to train and validate the model. It is equal to number of batches times n
        input_graph_clust_dic: a dictionary that maps the most frequent grapheme clusters to integers
        input_embedding_dim: length of the embedding vectors for each grapheme cluster
        input_hunits: number of units used in each cell of LSTM
        input_dropout_rate: dropout rate used in layers after the embedding and after the bidirectional LSTM
        input_output_dim: dimension of the output layer
        input_epochs: number of epochs used to train the model
        input_training_data: name of the data used to train the model
        input_evaluating_data: name of the data used to evaluate the model
        input_embedding_type: determines what type to be used in LSTM model
    """
    def __init__(self, input_name, input_n, input_t, input_clusters_num, input_embedding_dim, input_hunits,
                 input_dropout_rate, input_output_dim, input_epochs, input_training_data, input_evaluating_data,
                 input_language, input_embedding_type):
        self.name = input_name
        self.n = input_n
        self.t = input_t
        if self.t % self.n != 0:
            print("Warning: t is not divided by n")
        self.clusters_num = input_clusters_num
        self.batch_size = self.t // self.n  # number of batches used to train the model. It is defined as t // n
        self.embedding_dim = input_embedding_dim
        self.hunits = input_hunits
        self.dropout_rate = input_dropout_rate
        self.output_dim = input_output_dim
        self.epochs = input_epochs
        self.training_data = input_training_data
        self.evaluating_data = input_evaluating_data
        self.model = None
        self.language = input_language
        self.embedding_type = input_embedding_type

        # Constructing the grapheme cluster dictionary
        cnt = 0
        self.graph_clust_dic = dict()
        for key in constants.THAI_GRAPH_CLUST_RATIO.keys():
            if cnt < self.clusters_num - 1:
                self.graph_clust_dic[key] = cnt
            if cnt == self.clusters_num - 1:
                break
            cnt += 1

        # Constructing the letters dictionary to be used for generalized vectors
        if self.language == "Thai":
            smallest_unicode_dec = int("0E01", 16)
            largest_unicode_dec = int("0E5B", 16)
            self.letters_dic = dict()
            cnt = 0
            for i in range(smallest_unicode_dec, largest_unicode_dec + 1):
                ch = chr(i)
                if constants.THAI_CHAR_TYPE_TO_BUCKET[Char.charType(ch)] == 1:
                    self.letters_dic[ch] = cnt
                    cnt += 1

    def _get_trainable_data(self, input_line):
        """
        Given a segmented line, extracts x_data (with respect to a dictionary that maps grapheme clusters to integers)
        and y_data which is a n*4 matrix that represents BIES where n is the length of the unsegmented line. All grapheme
        clusters not found in the dictionary are set to the largest value of the dictionary plus 1
        Args:
            input_line: the unsegmented line
        """

        # Finding word breakpoints
        # Note that it is possible that input is segmented manually instead of icu, but for this function, we set that as
        # icu and set `man_segmented = None`, so the function works for both icu and manually segmented strings.
        line = Line(input_line, "icu_segmented")
        true_bies = line.get_bies("icu")

        # Making x_data and y_data
        y_data = true_bies.mat
        line_len = len(line.char_brkpoints) - 1

        x_data = []
        for i in range(line_len):
            char_start = line.char_brkpoints[i]
            char_finish = line.char_brkpoints[i + 1]
            curr_char = line.unsegmented[char_start: char_finish]
            x_data.append(GraphemeCluster(curr_char, self.graph_clust_dic, self.letters_dic))

        return x_data, y_data

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
            input_str = get_best_data_text(starting_text=1, ending_text=2, pseudo=False)
            x_data, y_data = self._get_trainable_data(input_str)
            if self.t > len(x_data):
                print("Warning: size of the training data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]

        elif self.training_data == "pseudo BEST":
            input_str = get_best_data_text(starting_text=1, ending_text=10, pseudo=True)
            x_data, y_data = self._get_trainable_data(input_str)
            if self.t > len(x_data):
                print("Warning: size of the training data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]

        elif self.training_data == "my":
            input_str = combine_lines_of_file("./Data/my_train.txt", input_type="unsegmented",
                                                  output_type="icu_segmented")
            x_data, y_data = self._get_trainable_data(input_str)
            if self.t > len(x_data):
                print("Warning: size of the training data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        else:
            print("Warning: no implementation for this training data exists!")
        train_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=self.batch_size,
                                              dim_output=self.output_dim)

        # Get validation data of length self.t
        if self.training_data == "BEST":
            input_str = get_best_data_text(starting_text=10, ending_text=12, pseudo=False)
            x_data, y_data = self._get_trainable_data(input_str)
            if self.t > len(x_data):
                print("Warning: size of the validation data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        elif self.training_data == "pseudo BEST":
            input_str = get_best_data_text(starting_text=10, ending_text=20, pseudo=True)
            x_data, y_data = self._get_trainable_data(input_str)
            if self.t > len(x_data):
                print("Warning: size of the validation data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        elif self.training_data == "my":
            input_str = combine_lines_of_file("./Data/my_valid.txt", input_type="unsegmented",
                                                  output_type="icu_segmented")
            x_data, y_data = self._get_trainable_data(input_str)
            if self.t > len(x_data):
                print("Warning: size of the training data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        else:
            print("Warning: no implementation for this validation data exists!")
        valid_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=self.batch_size,
                                              dim_output=self.output_dim)

        # Building the model
        model = Sequential()
        if self.embedding_type == "grapheme_clusters_tf":
            model.add(Embedding(input_dim=self.clusters_num, output_dim=self.embedding_dim, input_length=self.n))
        if self.embedding_type == "grapheme_clusters_man":
            model.add(TimeDistributed(Dense(input_dim=self.clusters_num, units=self.embedding_dim, use_bias=False,
                                            kernel_initializer='uniform')))
        if self.embedding_type == "generalized_vectors":
            model.add(TimeDistributed(Dense(self.embedding_dim, activation=None, use_bias=False,
                                            kernel_initializer='uniform')))
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(self.hunits, return_sequences=True), input_shape=(self.n, 1)))
        model.add(Dropout(self.dropout_rate))
        model.add(TimeDistributed(Dense(self.output_dim, activation='softmax')))
        opt = keras.optimizers.Adam(learning_rate=0.1)
        # opt = keras.optimizers.SGD(learning_rate=0.4, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Fitting the model
        model.fit(train_generator.generate(type=self.embedding_type), steps_per_epoch=self.t // self.batch_size,
                  epochs=self.epochs, validation_data=valid_generator.generate(type=self.embedding_type),
                  validation_steps=self.t // self.batch_size)
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
            input_str = get_best_data_text(starting_text=40, ending_text=45, pseudo=False)
            x_data, y_data = self._get_trainable_data(input_str)
        elif self.evaluating_data == "SAFT":
            input_str = combine_lines_of_file("./Data/SAFT/test.txt", input_type="man_segmented",
                                              output_type="man_segmented")
            x_data, y_data = self._get_trainable_data(input_str)
        elif self.evaluating_data == "my":
            input_str = combine_lines_of_file("./Data/my_test.txt", input_type="unsegmented",
                                              output_type="icu_segmented")
            x_data, y_data = self._get_trainable_data(input_str)
        else:
            print("Warning: no implementation for this evaluation data exists!")

        num_big_batches = len(x_data)//self.t
        accuracy = Accuracy()
        for k in range(num_big_batches-1):
            curr_x_data = x_data[k*self.t: (k+1)*self.t]
            curr_y_data = y_data[k * self.t: (k + 1) * self.t, :]
            test_generator = KerasBatchGenerator(curr_x_data, curr_y_data, n=self.n, batch_size=self.batch_size,
                                                 dim_output=self.output_dim)

            # Testing batch by batch (each batch of length self.n)
            all_test_input, all_actual_y = test_generator.generate_all(type=self.embedding_type)
            all_y_hat = self.model.predict(all_test_input, batch_size=self.batch_size)

            for i in range(self.batch_size):
                actual_y = all_actual_y[i, :, :]
                actual_y = Bies(input=actual_y, input_type="mat")
                y_hat = Bies(input=all_y_hat[i, :, :], input_type="mat")
                accuracy.update(true_bies=actual_y.str, est_bies=y_hat.str)

        print("The BIES accuracy in test_model function: {}".format(accuracy.get_bies_accuracy()))
        print("The F1 socre in test_model function: {}".format(accuracy.get_f1_score()))

    def _test_text_line_by_line(self, file, line_limit):
        """
        This function tests the model fitted in self.train() using BEST data set. Unlike test_model() function, this
        function tests the model line by line. It combines very short lines together before testing.
        Args:
            file: the address of the file that is going to be tested
            line_limit: number of lines to be tested
        """
        line_counter = 0
        accuracy = Accuracy()
        with open(file) as f:
            for line in f:
                if line_counter == line_limit:
                    break
                line = clean_line(line)
                if line == -1:
                    continue
                line_counter += 1

                # Get trainable data
                x_data, y_data = self._get_trainable_data(line)

                # Use the manual predict function
                y_hat = Bies(input=self._manual_predict(x_data), input_type="mat")
                y_hat.normalize_bies()

                actual_y = Bies(input=y_data, input_type="mat")
                accuracy.update(true_bies=actual_y.str, est_bies=y_hat.str)

            print("The BIES accuracy (line by line) for file {} : {}".format(file, accuracy.get_bies_accuracy()))
            print("The F1 score (line by line) for file {} : {}".format(file, accuracy.get_f1_score()))
            return accuracy

    def test_model_line_by_line(self):
        """
        This function uses the test_text_line_by_line() to test the model by a range of texts in BEST data set. The
        final score is the average of scores computed for each individual text.
        """
        accuracy = Accuracy()
        if self.evaluating_data == "BEST":
            category = ["news", "encyclopedia", "article", "novel"]
            for text_num in range(40, 45):
                print("testing text {}".format(text_num))
                for cat in category:
                    text_num_str = "{}".format(text_num).zfill(5)
                    file = "./Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt"
                    text_acc = self._test_text_line_by_line(file, line_limit=-1)
                    accuracy.merge_accuracy(text_acc)

        elif self.evaluating_data == "my":
            file = "./Data/my_test_segmented.txt"
            num_lines = sum(1 for _line in open(file))
            line_limit = 2000
            if line_limit > num_lines:
                print("Warning: number of lines you are using is larger than the total number of lines in " + file)
            text_acc = self._test_text_line_by_line(file, line_limit=line_limit)
            accuracy.merge_accuracy(text_acc)

        else:
            print("Warning: no implementation for this evaluation data exists!")
        print("The BIES accuracy by test_model_line_by_line function: {}".format(accuracy.get_bies_accuracy()))
        print("The F1 score by test_model_line_by_line function: {}".format(accuracy.get_f1_score()))

    def _manual_predict(self, test_input):
        """
        Implementation of the tf.predict function manually. This function works for inputs of any length, and only uses
        model weights obtained from self.model.weights.
        Args:
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
            if self.embedding_type == "grapheme_clusters_tf":
                input_graph_id = test_input[i].graph_clust_id
                x_t = embedarr[input_graph_id, :]
                x_t = x_t.reshape(1, x_t.shape[0])
            if self.embedding_type == "grapheme_clusters_man":
                input_graph_vec = test_input[i].graph_clust_vec
                x_t = input_graph_vec.dot(embedarr)
            if self.embedding_type == "generalized_vectors":
                input_generalized_vec = test_input[i].generalized_vec
                x_t = input_generalized_vec.dot(embedarr)
            h_fw, c_fw = self._compute_hc(weightLSTM, x_t, h_fw, c_fw)
            all_h_fw[i, :] = h_fw

        # Backward LSTM
        embedarr = self.model.weights[0]
        embedarr = embedarr.numpy()
        weightLSTM = self.model.weights[4: 7]
        c_bw = np.zeros([1, self.hunits])
        h_bw = np.zeros([1, self.hunits])
        all_h_bw = np.zeros([len(test_input), self.hunits])
        for i in range(len(test_input) - 1, -1, -1):
            if self.embedding_type == "grapheme_clusters_tf":
                input_graph_id = test_input[i].graph_clust_id
                x_t = embedarr[input_graph_id, :]
                x_t = x_t.reshape(1, x_t.shape[0])
            if self.embedding_type == "grapheme_clusters_man":
                input_graph_vec = test_input[i].graph_clust_vec
                x_t = input_graph_vec.dot(embedarr)
            if self.embedding_type == "generalized_vectors":
                input_generalized_vec = test_input[i].generalized_vec
                x_t = input_generalized_vec.dot(embedarr)
            h_bw, c_bw = self._compute_hc(weightLSTM, x_t, h_bw, c_bw)
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

    def _compute_hc(self, weight, x_t, h_tm1, c_tm1):
        """
        Given weights of a LSTM model, the input at time t, and values for h and c at time t-1, compute the values of h and
        c for time t.
        Args:
            weight: a list of three matrices, which are W (from input to cell), U (from h to cell), and b (bias) respectively.
            Dimensions: warr.shape = (embedding_dim, hunits*4), uarr.shape = (hunits, hunits*4), barr.shape = (hunits*4,)
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

    def segment_arbitrary_line(self, input_line):
        """
        This function uses the lstm model to segmenent an unsegmented line. It is intended to be used to analyze errors.
        Args:
            input_line: the unsegmented input line
        """
        line = Line(input_line, "unsegmented")
        line_len = len(line.char_brkpoints) - 1
        x_data = []
        excess_grapheme_ids = max(self.graph_clust_dic.values()) + 1
        for i in range(line_len):
            char_start = line.char_brkpoints[i]
            char_finish = line.char_brkpoints[i + 1]
            curr_char = line.unsegmented[char_start: char_finish]
            x_data.append(GraphemeCluster(curr_char, self.graph_clust_dic, self.letters_dic))
        y_hat = Bies(input=self._manual_predict(x_data), input_type="mat")
        y_hat.normalize_bies()

        y_hat = Bies(input=self._manual_predict(x_data), input_type="mat")
        y_hat.normalize_bies()
        print("Input line     : {}".format(line.unsegmented))
        print("ICU segmented  : {}".format(line.icu_segmented))
        print("LSTM segmented : {}".format(y_hat.str))

    def save_model(self):
        self.model.save("./Models/" + self.name)
        np.save(os.getcwd() + "/Models/" + self.name + "/" + "weights", self.model.weights)
        with open(os.getcwd() + "/Models/" + self.name + "/" + "weights.json", 'w') as wfile:
            output = dict()
            output["model"] = self.name
            output["dic"] = self.graph_clust_dic
            for i in range(len(self.model.weights)):
                dic_model = dict()
                dic_model["v"] = 1
                mat = self.model.weights[i].numpy()
                dim0 = mat.shape[0]
                dim1 = 1
                if len(mat.shape) == 1:
                    dic_model["dim"] = [dim0]
                else:
                    dim1 = mat.shape[1]
                    dic_model["dim"] = [dim0, dim1]
                serial_mat = np.reshape(mat, newshape=[1, dim0 * dim1])
                dic_model["data"] = serial_mat.tolist()[0]
                output["mat{}".format(i+1)] = dic_model
            json.dump(output, wfile)

    def set_model(self, input_model):
        self.model = input_model
