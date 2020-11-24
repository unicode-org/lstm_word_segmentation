from pathlib import Path
import numpy as np
import json
from icu import Char
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Embedding, Dropout
from tensorflow import keras

from . import constants
from .helpers import sigmoid
from .text_helpers import get_whole_file_segmented, get_best_data_text, get_lines_of_text
from .accuracy import Accuracy
from .line import Line
from .bies import Bies
from .grapheme_cluster import GraphemeCluster


class KerasBatchGenerator(object):
    """
    A batch generator component, which is used to generate batches for training, validation, and evaluation.
    Args:
        x_data: A list of GraphemeCluster objects that is the input of the model
        y_data: A np array that contains output of the model
        n: length of the input and output in each batch
        batch_size: number of batches
    """
    def __init__(self, x_data, y_data, n, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.batch_size = batch_size
        self.dim_output = self.y_data.shape[1]
        if len(x_data) != y_data.shape[0]:
            print("Warning: x_data and y_data have not compatible sizes!")
        if len(x_data) < batch_size * n:
            print("Warning: x_data or y_data is not large enough!")

    def generate(self, embedding_type):
        """
        This function generates batches used for training and validation
        """
        x, y = self.generate_once(embedding_type)
        while True:
            yield x, y

    def generate_once(self, embedding_type):
        """
        This function generates batches only once and is used for testing
        """
        y = np.zeros([self.batch_size, self.n, self.dim_output])
        x = None
        if embedding_type == "grapheme_clusters_tf":
            x = np.zeros([self.batch_size, self.n])
        elif embedding_type == "grapheme_clusters_man":
            x = np.zeros([self.batch_size, self.n, self.x_data[0].num_clusters])
        elif embedding_type == "generalized_vectors":
            x = np.zeros([self.batch_size, self.n, self.x_data[0].generalized_vec_length])
        else:
            print("Warning: the embedding type is not valid")
        for i in range(self.batch_size):
            for j in range(self.n):
                if embedding_type == "grapheme_clusters_tf":
                    x[i, j] = self.x_data[self.n*i + j].graph_clust_id
                if embedding_type == "grapheme_clusters_man":
                    x[i, j, :] = self.x_data[self.n*i + j].graph_clust_vec
                if embedding_type == "generalized_vectors":
                    x[i, j, :] = self.x_data[self.n*i + j].generalized_vec
            y[i, :, :] = self.y_data[self.n * i: self.n * (i + 1), :]
        return x, y


class WordSegmenter:
    """
    A class that let you make a bi-directional LSTM, train it, and test it.
    Args:
        input_n: Length of the input for LSTM model
        input_t: The total length of data used to train and validate the model. It is equal to number of batches times n
        input_clusters_num: number of top grapheme clusters used to train the model
        input_embedding_dim: length of the embedding vectors for each grapheme cluster
        input_hunits: number of hidden units used in each cell of LSTM
        input_dropout_rate: dropout rate used in layers after the embedding and after the bidirectional LSTM
        input_output_dim: dimension of the output layer
        input_epochs: number of epochs used to train the model
        input_training_data: name of the data used to train the model
        input_evaluating_data: name of the data used to evaluate the model
        input_language: shows what is the language used to train the model (e.g. Thai, Burmese, ...)
        input_embedding_type: determines what type of embedding to be used in the model. Possible values are
        "grapheme_clusters_tf", "grapheme_clusters_man", and "generalized_vectors"
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
        self.language = input_language
        self.embedding_type = input_embedding_type
        self.model = None

        # Constructing the grapheme cluster dictionary
        ratios = None
        if self.language == "Thai":
            ratios = constants.THAI_GRAPH_CLUST_RATIO
        elif self.language == "Burmese":
            ratios = constants.BURMESE_GRAPH_CLUST_RATIO
        else:
            print("Warning: the input language is not supported")
        cnt = 0
        self.graph_clust_dic = dict()
        for key in ratios.keys():
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
                if constants.CHAR_TYPE_TO_BUCKET[Char.charType(ch)] in [1, 2, 3]:
                    self.letters_dic[ch] = cnt
                    cnt += 1
        elif self.language == "Burmese":
            smallest_unicode_dec = int("1000", 16)
            largest_unicode_dec = int("109F", 16)
            self.letters_dic = dict()
            cnt = 0
            for i in range(smallest_unicode_dec, largest_unicode_dec + 1):
                ch = chr(i)
                if constants.CHAR_TYPE_TO_BUCKET[Char.charType(ch)] in [1, 2, 3]:
                    self.letters_dic[ch] = cnt
                    cnt += 1
        else:
            print("Warning: the grapheme_vectors embedding type is not supported for this language")
        print("number of letters slots in generalized_vectors embedding is {}".format(len(self.letters_dic)))

    def _get_trainable_data(self, input_line):
        """
        Given a segmented line, generates a list of input data (with respect to the embedding type) and a n*4 np array
        that represents BIES where n is the length of the unsegmented line.
        Args:
            input_line: the unsegmented line
        """
        # Finding word breakpoints
        # Note that it is possible that input is segmented manually instead of icu. However, for both cases we set that
        # input_type equal to "icu_segmented" because that doesn't affect performance of this function. This is done
        # we won't need unnecessary if/else for "man_segmented" and "icu_segmented" throughout rest of the code.
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
        This function trains the model using the dataset specified in the __init__ function. It combine all lines in
        the data set with a space between them and then divide this large string into batches of fixed length self.n.
        in reading files, if `pseudo` is True then we use icu segmented text instead of manually segmented texts to
        train the model.
        """
        # Get training data of length self.t
        input_str = None
        if self.training_data == "BEST":
            input_str = get_best_data_text(starting_text=1, ending_text=2, pseudo=False)
        elif self.training_data == "pseudo BEST":
            input_str = get_best_data_text(starting_text=1, ending_text=10, pseudo=True)
        elif self.training_data == "my":
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_train.txt')
            input_str = get_whole_file_segmented(file, input_type="unsegmented", output_type="icu_segmented")
        else:
            print("Warning: no implementation for this training data exists!")
        x_data, y_data = self._get_trainable_data(input_str)
        if self.t > len(x_data):
            print("Warning: size of the training data is less than self.t")
        x_data = x_data[:self.t]
        y_data = y_data[:self.t, :]
        train_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=self.batch_size)

        # Get validation data of length self.t
        if self.training_data == "BEST":
            input_str = get_best_data_text(starting_text=10, ending_text=12, pseudo=False)
        elif self.training_data == "pseudo BEST":
            input_str = get_best_data_text(starting_text=10, ending_text=20, pseudo=True)
        elif self.training_data == "my":
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_valid.txt')
            input_str = get_whole_file_segmented(file, input_type="unsegmented", output_type="icu_segmented")
        else:
            print("Warning: no implementation for this validation data exists!")
        x_data, y_data = self._get_trainable_data(input_str)
        if self.t > len(x_data):
            print("Warning: size of the validation data is less than self.t")
        x_data = x_data[:self.t]
        y_data = y_data[:self.t, :]
        valid_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=self.batch_size)

        # Building the model
        model = Sequential()
        if self.embedding_type == "grapheme_clusters_tf":
            model.add(Embedding(input_dim=self.clusters_num, output_dim=self.embedding_dim, input_length=self.n))
        elif self.embedding_type == "grapheme_clusters_man":
            model.add(TimeDistributed(Dense(input_dim=self.clusters_num, units=self.embedding_dim, use_bias=False,
                                            kernel_initializer='uniform')))
        elif self.embedding_type == "generalized_vectors":
            model.add(TimeDistributed(Dense(self.embedding_dim, activation=None, use_bias=False,
                                            kernel_initializer='uniform')))
        else:
            print("Warning: the embedding_type is not implemented")
        model.add(Dropout(self.dropout_rate))
        model.add(Bidirectional(LSTM(self.hunits, return_sequences=True), input_shape=(self.n, 1)))
        model.add(Dropout(self.dropout_rate))
        model.add(TimeDistributed(Dense(self.output_dim, activation='softmax')))
        opt = keras.optimizers.Adam(learning_rate=0.1)
        # opt = keras.optimizers.SGD(learning_rate=0.4, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Fitting the model
        model.fit(train_generator.generate(embedding_type=self.embedding_type),
                  steps_per_epoch=self.t // self.batch_size, epochs=self.epochs,
                  validation_data=valid_generator.generate(embedding_type=self.embedding_type),
                  validation_steps=self.t // self.batch_size)
        self.model = model

    def test_model(self):
        """
        This function tests the model fitted in self.train(). It first divide the whole test data into big batches each
        of size self.t, and then break down each big batch intor smaller batches each of size self.n.
        """
        # Get testing data
        input_str = None
        if self.evaluating_data == "BEST":
            input_str = get_best_data_text(starting_text=40, ending_text=45, pseudo=False)
        elif self.evaluating_data == "SAFT":
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT/test.txt')
            input_str = get_whole_file_segmented(file, input_type="man_segmented", output_type="man_segmented")
        elif self.evaluating_data == "my":
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_test.txt')
            input_str = get_whole_file_segmented(file, input_type="unsegmented", output_type="icu_segmented")
        else:
            print("Warning: no implementation for this evaluation data exists!")
        x_data, y_data = self._get_trainable_data(input_str)

        num_big_batches = len(x_data)//self.t
        if len(x_data) < self.t:
            print("Warning: length of the test data is smaller than self.t")
        accuracy = Accuracy()
        # Testing each big batch
        for k in range(num_big_batches):
            curr_x_data = x_data[k*self.t: (k+1)*self.t]
            curr_y_data = y_data[k * self.t: (k + 1) * self.t, :]
            test_generator = KerasBatchGenerator(curr_x_data, curr_y_data, n=self.n, batch_size=self.batch_size)

            # Testing each mini batch
            all_test_input, all_actual_y = test_generator.generate_once(embedding_type=self.embedding_type)
            all_y_hat = self.model.predict(all_test_input, batch_size=self.batch_size)
            for i in range(self.batch_size):
                actual_y = all_actual_y[i, :, :]
                actual_y = Bies(input_bies=actual_y, input_type="mat")
                y_hat = Bies(input_bies=all_y_hat[i, :, :], input_type="mat")
                accuracy.update(true_bies=actual_y.str, est_bies=y_hat.str)

        print("The BIES accuracy in test_model function: {}".format(accuracy.get_bies_accuracy()))
        print("The F1 socre in test_model function: {}".format(accuracy.get_f1_score()))
        return accuracy.get_bies_accuracy()

    def _test_text_line_by_line(self, file, line_limit):
        """
        This function tests the model fitted in self.train() using a single file textthat contains segmented lines.
        Unlike self.test_model(), this function test texts line by line.
        Args:
            file: the address of the file that is going to be tested
            line_limit: number of lines to be tested
        """
        lines = get_lines_of_text(file, "man_segmented")
        if len(lines) < line_limit:
            print("Warning: not enough lines in the test file")
        accuracy = Accuracy()
        for line in lines[:line_limit]:
            x_data, y_data = self._get_trainable_data(line.man_segmented)

            # Using the manual predict function for lines that are not necessarily self.n long
            y_hat = Bies(input_bies=self._manual_predict(x_data), input_type="mat")
            y_hat.normalize_bies()

            # Updating the accuracy using the new line
            actual_y = Bies(input_bies=y_data, input_type="mat")
            accuracy.update(true_bies=actual_y.str, est_bies=y_hat.str)
        print("The BIES accuracy (line by line) for file {} : {}".format(file, accuracy.get_bies_accuracy()))
        print("The F1 score (line by line) for file {} : {}".format(file, accuracy.get_f1_score()))
        return accuracy

    def test_model_line_by_line(self):
        """
        This function uses the evaluating data to test the model line by line.
        """
        accuracy = Accuracy()
        if self.evaluating_data == "BEST":
            if self.language != "Thai":
                print("Warning: the Best data is in Thai and you are testing a model in another language")
            category = ["news", "encyclopedia", "article", "novel"]
            for text_num in range(40, 45):
                print("testing text {}".format(text_num))
                for cat in category:
                    text_num_str = "{}".format(text_num).zfill(5)
                    file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/Best/{}/{}_".format(cat, cat)
                                         + text_num_str + ".txt")
                    text_acc = self._test_text_line_by_line(file, line_limit=-1)
                    accuracy.merge_accuracy(text_acc)

        elif self.evaluating_data == "my":
            if self.language != "Burmese":
                print("Warning: the my.text data is in Burmese and you are testing a model in another language")
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_test_segmented.txt')
            text_acc = self._test_text_line_by_line(file, line_limit=1000)
            accuracy.merge_accuracy(text_acc)
        elif self.evaluating_data == "SAFT":
            if self.language != "Thai":
                print("Warning: the current SAFT data is in Thai and you are testing a model in another language")
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT/test.txt')
            text_acc = self._test_text_line_by_line(file, line_limit=-1)
            accuracy.merge_accuracy(text_acc)
        else:
            print("Warning: no implementation for line by line evaluating this data exists")
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
            x_t = None
            if self.embedding_type == "grapheme_clusters_tf":
                input_graph_id = test_input[i].graph_clust_id
                x_t = embedarr[input_graph_id, :]
                x_t = x_t.reshape(1, x_t.shape[0])
            elif self.embedding_type == "grapheme_clusters_man":
                input_graph_vec = test_input[i].graph_clust_vec
                x_t = input_graph_vec.dot(embedarr)
            elif self.embedding_type == "generalized_vectors":
                input_generalized_vec = test_input[i].generalized_vec
                x_t = input_generalized_vec.dot(embedarr)
            else:
                print("Warning: this embedding type is not implemented")
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
            x_t = None
            if self.embedding_type == "grapheme_clusters_tf":
                input_graph_id = test_input[i].graph_clust_id
                x_t = embedarr[input_graph_id, :]
                x_t = x_t.reshape(1, x_t.shape[0])
            elif self.embedding_type == "grapheme_clusters_man":
                input_graph_vec = test_input[i].graph_clust_vec
                x_t = input_graph_vec.dot(embedarr)
            elif self.embedding_type == "generalized_vectors":
                input_generalized_vec = test_input[i].generalized_vec
                x_t = input_generalized_vec.dot(embedarr)
            else:
                print("Warning: this embedding type is not implemented")
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
        Given weights of a LSTM model, the input at time t, and values for h and c at time t-1, this function compute
        the values of h and c at time t.
        Args:
            weight: a list of three matrices, which are W (from input to cell), U (from h to cell), and b (bias) respectively.
            x_t: the input at time t
            h_tm1: value of h for time t-1
            c_tm1: value of c for time t-1
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
        This function uses the LSTM model to segment an unsegmented line and compare it to ICU and deepcut.
        Args:
            input_line: the unsegmented input line
        """
        line = Line(input_line, "unsegmented")
        line_len = len(line.char_brkpoints) - 1

        # Making the input to and the output from the lstm model
        x_data = []
        for i in range(line_len):
            char_start = line.char_brkpoints[i]
            char_finish = line.char_brkpoints[i + 1]
            curr_char = line.unsegmented[char_start: char_finish]
            x_data.append(GraphemeCluster(curr_char, self.graph_clust_dic, self.letters_dic))
        y_hat = Bies(input_bies=self._manual_predict(x_data), input_type="mat")
        y_hat.normalize_bies()

        # Making a pretty version of the output of the LSTM, where bars shown the boundary of segmented words
        y_hat_pretty = ""
        for i in range(line_len):
            char_start = line.char_brkpoints[i]
            char_finish = line.char_brkpoints[i + 1]
            curr_char = line.unsegmented[char_start: char_finish]
            if y_hat.str[i] in ['b', 's']:
                y_hat_pretty += "|"
            y_hat_pretty += curr_char
        y_hat_pretty += "|"

        # Showing the output
        print("Input line        : {}".format(line.unsegmented))
        print("ICU segmented     : {}".format(line.icu_segmented))
        print("LSTM segmented    : {}".format(y_hat_pretty))
        print("Deepcut segmented : {}".format(line.get_deepcut_segmented()))

    def save_model(self):
        """
        This function saves the current trained model of this word_segmenter instance.
        """
        # Save the model using Keras
        self.model.save(Path.joinpath(Path(__file__).parent.parent.absolute(), "Models/" + self.name))
        # Save one np array that holds all weights
        file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Models/" + self.name + "/weights")
        np.save(str(file), self.model.weights)

        # Save the model in json format, that has both weights and grapheme clusters dictionary
        json_file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Models/" + self.name + "/weights.json")
        with open(str(json_file), 'w') as wfile:
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
        """
        This function set the current model to an input model
        input_model: the input model
        """
        self.model = input_model
