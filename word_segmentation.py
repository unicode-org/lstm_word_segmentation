# coding=utf-8
import numpy as np
import os
from icu import BreakIterator, Locale
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
from bayes_opt import BayesianOptimization
import json
import pickle
import collections


################# basic helper functions #################

def is_ascii(input_str):
    """
    A very basic function that checks if all elements of str are ASCII or not
    Args:
        input_str: input string
    """
    return all(ord(char) < 128 for char in input_str)


def diff_strings(str1, str2):
    """
    A function that returns the number of elements of two strings that are not identical
    Args:
        str1: the first string
        str2: the second string
    """
    if len(str1) != len(str2):
        print("Warning: length of two strings are not equal")
    return sum(str1[i] != str2[i] for i in range(len(str1)))


def sigmoid(x):
    """
    Computes the sigmoid function for a scalar
    Args:
        x: the scalar
    """
    return 1.0/(1.0+np.exp(-x))


################# functions for handling text files #################

def add_additional_bars(read_filename, write_filename):
    """
    This function reads a segmented file and add bars around each space in it. It assumes that spaces are used as
    breakpoints in the segmentation (just like "|")
    Args:
        read_filename: Address of the input file
        write_filename: Address of the output file
    """
    wfile = open(write_filename, 'w')
    with open(read_filename) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            new_line = ""
            for i in range(len(line)):
                ch = line[i]
                # If you want to put lines bars around punctuations as well, you should use comment the {if ch == " "}
                # and uncomment {if 32 <= ord(ch) <= 47 or 58 <= ord(ch) <= 64}.
                # The later if puts bars for !? as |!||?| instead of |!|?|. This should be fixed if the
                # following if is going to be used. It can easily be fixed by keeping track of the last character in
                # new_line.
                # if 32 <= ord(ch) <= 47 or 58 <= ord(ch) <= 64:
                if ch == " ":
                    if i == 0:
                        if i+1 < len(line) and line[i+1] == "|":
                            new_line = new_line + "|" + ch
                        else:
                            new_line = new_line + "|" + ch + "|"
                    elif i == len(line)-1:
                        if line[i-1] == "|":
                            new_line = new_line + ch + "|"
                        else:
                            new_line = new_line + "|" + ch + "|"
                    else:
                        if line[i-1] != "|" and line[i+1] != "|":
                            new_line = new_line + "|" + ch + "|"
                        if line[i-1] == "|" and line[i+1] != "|":
                            new_line = new_line + ch + "|"
                        if line[i-1] != "|" and line[i+1] == "|":
                            new_line = new_line + "|" + ch
                        if line[i-1] == "|" and line[i+1] == "|":
                            new_line = new_line + ch
                else:
                    new_line += ch
            new_line += "\n"
            wfile.write(new_line)


def combine_lines_of_file(filename, input_type, output_type):
    """
    This function first combine all lines in a file where each two lines are separated with a space, and then uses ICU
    to segment the new long string.
    Note: Because in some of the Burmese texts some lines start with code points that are not valid, I first combine all
    lines and then segment them, rather than first segmenting each line and then combining them. This can result in a
    more robust segmentation. Eample: see line 457457 of the my_train.txt
    Args:
        filename: address of the input file
        input_type: determines if the input is unsegmented, manually segmented, or ICU segmented
        output_type: determines if we want the output to be unsegmented, manually segmented, or ICU segmented
    """
    all_file_line = ""
    with open(filename) as f:
        for file_line in f:
            file_line = file_line.strip()
            if is_ascii(file_line):
                continue
            if len(all_file_line) == 0:
                all_file_line = file_line
            else:
                all_file_line = all_file_line + " " + file_line
    all_file_line = Line(all_file_line, input_type)
    if output_type == "man_segmented":
        return all_file_line.man_segmented
    if output_type == "icu_segmented":
        return all_file_line.icu_segmented
    if output_type == "unsegmented":
        return all_file_line.unsegmented


def get_best_data_text(starting_text, ending_text, pseudo):
    """
    Gives a long string, that contains all lines (separated by a single space) from BEST data with numbers in a range
    This function uses data from all sources (news, encyclopedia, article, and novel)
    It removes all texts between pair of tags such as (<NE>, </NE>), assures that the string starts and ends with "|",
    and ignores empty lines, lines with "http" in them, and lines that are all in ascii (since these are not segmented
    in the BEST data set)
    Args:
        starting_text: number or the smallest text
        ending_text: number or the largest text + 1
        pseudo: if True, it means we use pseudo segmented data, if False, we use BEST segmentation
    """
    category = ["news", "encyclopedia", "article", "novel"]
    out_str = ""
    for text_num in range(starting_text, ending_text):
        for cat in category:
            text_num_str = "{}".format(text_num).zfill(5)
            file = "./Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt"
            with open(file) as f:
                for file_line in f:
                    file_line = clean_line(file_line)
                    if file_line == -1:
                        continue
                    line = Line(file_line, "man_segmented")

                    # If pseudo is True then unsegment the text and re-segment it using ICU
                    new_line = line.man_segmented
                    if pseudo:
                        new_line = line.icu_segmented

                    if len(out_str) == 0:
                        out_str = new_line
                    else:
                        out_str = out_str + " " + new_line
    return out_str


def divide_train_test_data(input_text, train_text, valid_text, test_text):
    """
    This function divides a file into three new files, that contain train data, validation data, and testing data
    Args:
        input_text: address of the original file
        train_text: address to store the train data in it
        valid_text: address to store the validation data in it
        test_text: address to store the test file in it
    """
    train_ratio = 0.4
    valid_ratio = 0.4
    bucket_size = 20
    train_file = open(train_text, 'w')
    valid_file = open(valid_text, 'w')
    test_file = open(test_text, 'w')
    line_counter = 0
    with open(input_text) as f:
        for line in f:
            line_counter += 1
            line = line.strip()
            if is_ascii(line):
                continue
            if line_counter % bucket_size <= bucket_size*train_ratio:
                train_file.write(line + "\n")
            elif bucket_size*train_ratio < line_counter % bucket_size <= bucket_size*(train_ratio+valid_ratio):
                valid_file.write(line + "\n")
            else:
                test_file.write(line + "\n")


def store_icu_segmented_file(unseg_filename, seg_filename):
    """
    This function uses ICU to segment a file line by line and store that segmented file. The lines in the input file
    must be unsegmented.
    Args:
        unseg_filename: address of the unsegmented file
        seg_filename: address that the segmented file will be stored
    """
    wfile = open(seg_filename, 'w')
    with open(unseg_filename) as f:
        for file_line in f:
            file_line = file_line.strip()
            if len(file_line) == 0:
                continue
            line = Line(file_line, "unsegmented")
            wfile.write(line.icu_segmented + "\n")


def remove_tags(line, st_tag, fn_tag):
    """
    Given a string and two substrings, remove any text between these tags.
    It handles spaces around tags as follows:
        abc|<NE>def</NE>|ghi      ---> abc|ghi
        abc| |<NE>def</NE>|ghi    ---> abc| |ghi
        abc|<NE>def</NE>| |ghi    ---> abc| |ghi
        abc| |<NE>def</NE>| |ghi  ---> abc| |ghi
    Args:
        line: the input string
        st_tag: the first substring
        fn_tag: the secibd substring
    """

    new_line = ""
    st_ind = 0
    while st_ind < len(line):
        curr_is_tag = False
        if line[st_ind: st_ind+len(st_tag)] == st_tag:
            curr_is_tag = True
            fn_ind = st_ind
            while fn_ind < len(line):
                if line[fn_ind: fn_ind+len(fn_tag)] == fn_tag:
                    fn_ind = fn_ind+len(fn_tag) + 1
                    if st_ind - 2 >= 0 and fn_ind+2 <= len(line):
                        if line[st_ind-2:st_ind] == " |" and line[fn_ind:fn_ind+2] == " |":
                            fn_ind += 2
                    st_ind = fn_ind
                    break
                else:
                    fn_ind += 1
        if st_ind < len(line):
            new_line += line[st_ind]
        if not curr_is_tag:
            st_ind += 1
    return new_line


def clean_line(line):
    """
    This line cleans a line as follows such that it is ready for process by different components of the code. It returns
    the clean line or -1, if the line should be omitted.
        1) remove tags and https from the line.
        2) Put a | at the begining and end of the line if it isn't already there
        3) if line is very short (len < 3) or if it is all in English or it has a link in it, return -1
    Args:
        line: the input line
    """
    line = line.strip()

    # Remove lines with links
    if "http" in line or len(line) == 0:
        return -1

    # Remove texts between following tags
    line = remove_tags(line, "<NE>", "</NE>")
    line = remove_tags(line, "<AB>", "</AB>")
    line = remove_tags(line, "<POEM>", "</POEM>")
    line = remove_tags(line, "<NER>", "</NER>")

    # Remove lines that are all fully in English
    if is_ascii(line):
        return -1

    # Add "|" to the end of each line if it is not there
    if len(line) >= 1 and line[len(line) - 1] != '|':
        line += "|"

    # Adding "|" to the start of each line if it is not there
    if line[0] != '|':
        line = '|' + line

    return line


################# functions for processing texts #################

def preprocess_thai(verbose):
    """
    This function uses the BEST data set to
        1) compute the grapheme cluster dictionary that holds the frequency of different grapheme clusters
        2) demonstrate the performance of icu word breakIterator and compute its accuracy
    Args:
        verbose: shows if we want to see how the algorithm is working or not
    """
    grapheme_clusters_dic = collections.defaultdict(int)
    icu_mismatch = 0
    icu_total_bies_lengths = 0
    correctly_segmented_words = 0
    true_words = 0
    segmented_words = 0
    for cat in ["news", "encyclopedia", "article", "novel"]:
        for text_num in range(1, 96):
            text_num_str = "{}".format(text_num).zfill(5)
            file = "./Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt"
            line_counter = 0
            with open(file) as f:
                for file_line in f:
                    line_counter += 1
                    file_line = clean_line(file_line)
                    if file_line == -1:
                        continue
                    line = Line(file_line, "man_segmented")

                    # Storing the grapheme clusters and their frequency in the dictionary
                    for i in range(len(line.char_brkpoints) - 1):
                        new_graph_clust = line.unsegmented[line.char_brkpoints[i]: line.char_brkpoints[i + 1]]
                        grapheme_clusters_dic[new_graph_clust] += 1

                    # Computing BIES corresponding to the manually segmented and ICU segmented
                    true_bies = get_bies(line.char_brkpoints, line.man_word_brkpoints)
                    true_bies_str = get_bies_string_from_softmax(np.transpose(true_bies))
                    icu_bies = get_bies(line.char_brkpoints, line.icu_word_brkpoints)
                    icu_bies_str = get_bies_string_from_softmax(np.transpose(icu_bies))

                    # Counting the number of mismatches between icu_bies and true_bies
                    icu_total_bies_lengths += len(icu_bies_str)
                    icu_mismatch += diff_strings(true_bies_str, icu_bies_str)

                    # Computing the F1-score between icu_bies and true_bies
                    f1_output = compute_f1_score(true_bies=true_bies_str, est_bies=icu_bies_str)
                    correctly_segmented_words += f1_output[1]
                    true_words += f1_output[2]
                    segmented_words += f1_output[3]

                    # Demonstrate how icu segmenter works
                    if verbose:
                        print("Cat: {}, Text number: {}, Line: {}".format(cat, text_num, line_counter))
                        line.display()
                        print("true bies string  : {}".format(true_bies_str))
                        print("icu bies string   : {}".format(icu_bies_str))
                        print('**********************************************************************************')
                    line_counter += 1

    icu_bies_accuracy = 1 - icu_mismatch/icu_total_bies_lengths

    precision = correctly_segmented_words / segmented_words
    recall = correctly_segmented_words / true_words
    icu_f1_accuracy = 0
    if precision + recall != 0:
        icu_f1_accuracy = 2 * precision * recall / (precision + recall)
    graph_clust_freq = grapheme_clusters_dic
    graph_clust_freq = {k: v for k, v in sorted(graph_clust_freq.items(), key=lambda item: item[1], reverse=True)}
    graph_clust_ratio = graph_clust_freq
    total = sum(graph_clust_ratio.values(), 0.0)
    graph_clust_ratio = {k: v / total for k, v in graph_clust_ratio.items()}
    return graph_clust_ratio, icu_bies_accuracy, icu_f1_accuracy


def preprocess_burmese(verbose):
    """
    This function uses the Google corpus crawler Burmese data set to
        1) Clean it by removing tabs from start of it
        1) Compute the grapheme cluster dictionary that holds the frequency of different grapheme clusters
        2) Demonstrate the performance of icu word breakIterator and compute its accuracy
    Args:
        verbose: shows if we want to see how the algorithm is working or not
    """
    chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getRoot())
    grapheme_clusters_dic = collections.defaultdict(int)
    file = "./Data/my.txt"
    with open(file) as f:
        line_counter = 0
        for file_line in f:
            file_line = file_line.strip()
            # If the resulting line is in ascii (including an empty line) continue
            if is_ascii(file_line):
                continue
            # Making the grapheme clusters brkpoints
            line = Line(file_line, "unsegmented")

            # Storing the grapheme clusters and their frequency in the dictionary
            for i in range(len(line.char_brkpoints) - 1):
                new_graph_clust = line.unsegmented[line.char_brkpoints[i]: line.char_brkpoints[i + 1]]
                grapheme_clusters_dic[new_graph_clust] += 1

            # Compute segmentations of icu and BIES associated with it
            icu_bies = get_bies(line.char_brkpoints, line.icu_word_brkpoints)

            # Demonstrate how icu segmenter works
            if verbose:
                line.display()
                icu_bies_str = get_bies_string_from_softmax(np.transpose(icu_bies))
                print("icu bies string   : {}".format(icu_bies_str))
                print('**********************************************************************************')
            line_counter += 1
    graph_clust_freq = grapheme_clusters_dic
    graph_clust_freq = {k: v for k, v in sorted(graph_clust_freq.items(), key=lambda item: item[1], reverse=True)}
    graph_clust_ratio = graph_clust_freq
    total = sum(graph_clust_ratio.values(), 0.0)
    graph_clust_ratio = {k: v / total for k, v in graph_clust_ratio.items()}
    return graph_clust_ratio


def get_bies(char_brkpoints, word_brkpoints):
    """
    Given break points for words and grapheme clusters, returns the matrix that represents BIES.
    The output is a matrix of size (n * 4) where n is the number of grapheme clusters in the string
    Args:
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
    Args:
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


def get_segmented_string_from_bies(line, bies):
    """
    This function gets an unsegmented line of text and BIES associated with it, to produce a segmented line of text
    where word boundaries are marked by "|"
    Args:
        line: a Line instance that stores the unsegmented line of text
        bies: BIES string associated with line
    """
    word_brkpoints = []
    for i in range(len(bies)):
        if bies[i] in ['b', 's']:
            word_brkpoints.append(line.char_brkpoints[i])
    word_brkpoints.append(line.char_brkpoints[-1])
    out = "|"
    for i in range(len(word_brkpoints) - 1):
        out += line.unsegmented[word_brkpoints[i]: word_brkpoints[i + 1]] + "|"
    return out


def compute_f1_score(true_bies, est_bies):
    """
    This function computes the F1-score given two BIES strings. One of these two strings indicates the true segmentation
     and the other one indicates the output of a word segmentation algorithm.
    Args:
        true_bies: BIES string corresponding to the true segmentation
        est_bies: BIES string corresponding to output of a word segmentation algorithm
    """
    true_word_brkpoints = []
    for i in range(len(true_bies)):
        if true_bies[i] in ['b', 's']:
            true_word_brkpoints.append(i)
    true_word_brkpoints.append(len(true_bies))
    est_word_brkpoints = []
    for i in range(len(est_bies)):
        if est_bies[i] in ['b', 's']:
            est_word_brkpoints.append(i)
    est_word_brkpoints.append(len(est_bies))
    ind0 = 0
    correctly_segmented_words = 0
    for i in range(len(true_word_brkpoints)-1):
        word_start = true_word_brkpoints[i]
        word_finish = true_word_brkpoints[i+1]
        # print("st = {}, fn = {}, count = {}".format(word_start, word_finish, correctly_segmented_words))
        while ind0 < len(est_word_brkpoints) and est_word_brkpoints[ind0] < word_start:
            ind0 += 1
        if ind0 > len(est_word_brkpoints):
            break
        elif est_word_brkpoints[ind0] > word_start:
            continue
        elif est_word_brkpoints[ind0] == word_start and est_word_brkpoints[ind0+1] == word_finish:
            correctly_segmented_words += 1
    true_words = len(true_word_brkpoints) - 1
    segmented_words = len(est_word_brkpoints)-1
    precision = correctly_segmented_words/segmented_words
    recall = correctly_segmented_words/true_words
    f1 = 0
    if precision+recall != 0:
        f1 = 2*precision*recall/(precision + recall)
    return f1, correctly_segmented_words, true_words, segmented_words


def compute_icu_accuracy(filename):
    """
    This function uses a dataset with segmented lines to compute the accuracy of icu word breakIterator
    Args:
        filename: The path of the file
    """
    line_counter = 0
    icu_mismatch = 0
    icu_total_bies_lengths = 0
    correctly_segmented_words = 0
    true_words = 0
    segmented_words = 0
    with open(filename) as f:
        for file_line in f:
            file_line = clean_line(file_line)
            if file_line == -1:
                continue
            line = Line(file_line, "man_segmented")
            true_bies = get_bies(line.char_brkpoints, line.man_word_brkpoints)
            true_bies_str = get_bies_string_from_softmax(np.transpose(true_bies))
            icu_bies = get_bies(line.char_brkpoints, line.icu_word_brkpoints)
            icu_bies_str = get_bies_string_from_softmax(np.transpose(icu_bies))

            # Counting the number of mismatches between icu_bies and true_bies
            icu_total_bies_lengths += len(icu_bies_str)
            icu_mismatch += diff_strings(true_bies_str, icu_bies_str)
            line_counter += 1

            # Computing the F1-score between icu_bies and true_bies
            f1_output = compute_f1_score(true_bies=true_bies_str, est_bies=icu_bies_str)
            correctly_segmented_words += f1_output[1]
            true_words += f1_output[2]
            segmented_words += f1_output[3]

        icu_bies_accuracy = 1 - icu_mismatch / icu_total_bies_lengths
        precision = correctly_segmented_words / segmented_words
        recall = correctly_segmented_words / true_words
        icu_f1_accuracy = 0
        if precision + recall != 0:
            icu_f1_accuracy = 2 * precision * recall / (precision + recall)
        return icu_bies_accuracy, icu_f1_accuracy


def normalize_bies(bies_str):
    """
    This function normalizes the input bies string to generate a bies string that makes sense. For example the output
    won't have substring such as "biiis", "biese" or "siie"
    Args:
        bies_str: The input bies string
    """
    if len(bies_str) == 0:
        return 's'
    out_bies = ""
    start_of_word = True
    # print(len(bies_str))

    for i in range(len(bies_str)):
        if start_of_word:
            if i == len(bies_str) - 1 or bies_str[i] == 's':
                # print("here at {}".format(i))
                out_bies += 's'
                start_of_word = True
                continue
            elif bies_str[i] == 'b':
                out_bies += bies_str[i]
                start_of_word = False
                continue
            elif bies_str[i] in ['i', 'e']:
                if bies_str[i+1] in ['i', 'e']:
                    out_bies += 'b'
                    start_of_word = False
                    continue
                else:
                    out_bies += 's'
                    start_of_word = True
                    continue
        if not start_of_word:
            if bies_str[i] == 'i':
                if i == len(bies_str)-1 or bies_str[i+1] in ['b', 's']:
                    out_bies += 'e'
                    start_of_word = True
                    continue
                else:
                    out_bies += bies_str[i]
                    start_of_word = False
                    continue
            if bies_str[i] == 'e':
                out_bies += bies_str[i]
                start_of_word = True
                continue
            if bies_str[i] in ['b', 's']:
                # In this case based on the previous if, the previous character must be b
                if i == len(bies_str)-1 or bies_str[i+1] in ['b', 's']:
                    out_bies += 'e'
                    start_of_word = True
                    continue
                else:
                    out_bies += 'i'
                    start_of_word = False
                    continue
    return out_bies


################################ functions for fitting LSTM ################################

def get_trainable_data(input_line, graph_clust_ids):
    """
    Given a segmented line, extracts x_data (with respect to a dictionary that maps grapheme clusters to integers)
    and y_data which is a n*4 matrix that represents BIES where n is the length of the unsegmented line. All grapheme
    clusters not found in the dictionary are set to the largest value of the dictionary plus 1
    Args:
        input_line: the unsegmented line
        graph_clust_ids: a dictionary that stores maps from grapheme clusters to integers
    """
    # Finding word breakpoints
    # Note that it is possible that input is segmented manually instead of icu, but for this function, we set that as
    # icu and set `man_segmented = None`, so the function works for both icu and manually segmented strings.
    line = Line(input_line, "icu_segmented")
    true_bies = get_bies(line.char_brkpoints, line.icu_word_brkpoints)

    # Making x_data and y_data
    line_len = len(line.char_brkpoints)-1
    x_data = np.zeros(shape=[line_len, 1])
    y_data = np.zeros(shape=[line_len, 4])
    excess_grapheme_ids = max(graph_clust_ids.values()) + 1
    for i in range(line_len):
        char_start = line.char_brkpoints[i]
        char_finish = line.char_brkpoints[i + 1]
        curr_char = line.unsegmented[char_start: char_finish]
        x_data[i, 0] = graph_clust_ids.get(curr_char, excess_grapheme_ids)
        y_data[i, :] = true_bies[:, i]
    return x_data, y_data


def compute_hc(weight, x_t, h_tm1, c_tm1):
    """
    Given weights of a LSTM model, the input at time t, and values for h and c at time t-1, compute the values of h and
    c for time t.
    Args:
        weights: a list of three matrices, which are W (from input to cell), U (from h to cell), and b (bias) respectively.
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


def lstm_score(hunits, embedding_dim):
    """
    Given the number of LSTM cells and embedding dimension, this function computes a score for a bi-directional LSTM
    model which is basically the accuracy of the model minus a weighted penalty function linear in number of parameters
    Args:
        hunits: number of LSTM cells in bi-directional LSTM model
        embedding_dim: length of output of the embedding layer
    """
    hunits = int(round(hunits))
    embedding_dim = int(round(embedding_dim))
    word_segmenter = WordSegmenter(input_n=50, input_t=100000, input_graph_clust_dic=graph_clust_dic,
                                   input_embedding_dim=embedding_dim, input_hunits=hunits, input_dropout_rate=0.2,
                                   input_output_dim=4, input_epochs=3, input_training_data="BEST",
                                   input_evaluating_data="BEST")
    word_segmenter.train_model()
    fitted_model = word_segmenter.get_model()
    lam = 1/88964  # This is number of parameters in the largest model
    C = 0
    return word_segmenter.test_model() - C * lam * fitted_model.count_params()


def perform_bayesian_optimization(hunits_lower, hunits_upper, embedding_dim_lower, embedding_dim_upper):
    """
    This function implements Bayesian optimization to search in a range of possible values for number of LSTM cells and
    embedding dimension to find the most accurate and parsimonious model. It uses the function LSTM_score to compute
    score of each model.
    Args:
        hunits_lower and hunits_upper: lower and upper bound of search region for number of LSTM cells
        embedding_dim_lower and embedding_dim_upper: lower and upper bound of search region for embedding dimension
    """
    bounds = {'hunits': (hunits_lower, hunits_upper), 'embedding_dim': (embedding_dim_lower, embedding_dim_upper)}
    optimizer = BayesianOptimization(
        f=lstm_score,
        pbounds=bounds,
        random_state=1,
    )
    optimizer.maximize(init_points=2, n_iter=10)
    print(optimizer.max)
    print(optimizer.res)


def convert_dic_to_json(dic):
    """
    Given a dictionary of characters, this function stores that dictionary in json format. It treats the special
    character " appropriately.
    Args:
        dic: the dictionary that needs to be stored in json format
    """
    out = "{"
    cnt = 0
    for k in dic.keys():
        if k == '"':
            out += "\"\\\"\": " + str(cnt)
        else:
            out += "\"" + k + "\": " + str(cnt)
        if cnt != len(dic.keys())-1:
            out += ", "
        cnt += 1
    out += "}"
    return out


################################ functions for converting to json ################################

def write_model_json(model_name, graph_clust_dic, model):
    """
    This function stores the LSTM model along the grapheme cluster dictionary in json format.
    Args:
        model_name: name of the model
        graph_clust_dic: the dictionary that stores all the grapheme clusters
        model: a tf model that has all the weights in it
    """
    with open(os.getcwd() + "/Models/" + model_name + "/" + "weights.json", 'w') as wfile:
        wfile.write("{\n")
        wfile.write("    " + "\"model\": \"" + model_name + "\",\n")
        dic_str = convert_dic_to_json(graph_clust_dic)
        wfile.write("    " + "\"dic\": " + dic_str + ",\n")
        for i in range(len(model.weights)):
            mat = model.weights[i].numpy()
            # mat = np.array([[1, 2], [3, 4]])
            dim0 = mat.shape[0]
            if len(mat.shape) == 1:
                dim1 = 1
            else:
                dim1 = mat.shape[1]
            wfile.write("    " + "\"mat{}\"".format(i + 1) + ": {\n")
            wfile.write("      \"v\": 1,\n")
            if len(mat.shape) == 1:
                wfile.write("      \"dim\": [" + str(dim0) + "],\n")
            else:
                wfile.write("      \"dim\": [" + str(dim0) + ", " + str(dim1) + "],\n")
            wfile.write("      \"data\": [")
            serial_mat = np.reshape(mat, newshape=[1, dim0 * dim1])
            for j in range(serial_mat.shape[1]):
                x = serial_mat[0, j]
                wfile.write(str(x))
                if j != serial_mat.shape[1] - 1:
                    wfile.write(", ")
            wfile.write("]\n")
            if i != len(model.weights)-1:
                wfile.write("    },\n")
            else:
                wfile.write("    }\n")
        wfile.write("}")


def write_grapheme_clusters_dic_json(graph_clust_ratio, graph_thrsh):
    """
    This function is intended to store the top grapheme clusters in the most compressed json format. It only list these
    grapheme clusters in a single line without any space between them. The ICU algorithm must be used later on the
    output of this function to detect different grapheme clusters. There is a small bug now in how to store grapheme "
    which needs to be fixed.
    Args:
        graph_clust_ratio: a dictionary that has all grapheme clusters sorted based on how frequent they appear
        graph_thrsh: the number of top grapheme clusters that we want to store
    """
    with open(os.getcwd() + "/Data/Thai_graph_clust_dic.json", 'w') as wfile:
        wfile.write("\"")
        cnt = 0
        for key in graph_clust_ratio.keys():
            cnt += 1
            if cnt > graph_thrsh:
                continue
            if key == "\"":
                wfile.write('\"')
            wfile.write(key)
        wfile.write("\"")


################################ functions for presentation ################################

def print_grapheme_clusters(ratios, thrsh):
    """
    This function analyzes the grapheme clusters, to see what percentage of them form which percent of the text, and
    provides a histogram that shows frequency of grapheme clusters
    ratios: a dictionary that holds the ratio of text that is represented by each grapheme cluster
    Args:
        ratios: a dictionary that shows the percentage of each grapheme cluster
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


################################ Classes ################################

class Line:
    """
    A class that stores different verions of a line: unsegmented, ICU segmented, and manually segmented (if exists).
    Args:
        unsegmented: the unsegmented version of the line
        icu_segmented: the ICU segmented version of the line
        man_segmented: the manually segmented version of the line (doesn't always exist)
        icu_word_brkpoints: a list of word break points computed by ICU
        man_word_brkpoints: a list of word break points computed manually (doesn't always exist)
        char_brkpoints: a list of exgtended grapheme cluster breakpoints for the line
    """
    def __init__(self, input_line, input_type):
        """
        The __init__ function creates a new instance of the class based on the input line and its type.
        Args:
            input_line: the input line. It should be a clean line in one language.
            input_type: determines what is the type of the input line. It can be segmented, icu_segmented, or
            man_segmented.
        """
        if input_type == "unsegmented":
            self.unsegmented = input_line
            self.compute_char_brkpoints()
            self.compute_icu_segmented()
            self.icu_word_brkpoints = self.compute_word_brkpoints(self.icu_segmented)
            self.man_segmented = None
            self.man_word_brkpoints = None

        if input_type == "icu_segmented":
            self.icu_segmented = input_line
            self.icu_word_brkpoints = self.compute_word_brkpoints(self.icu_segmented)
            self.unsegmented = input_line.replace("|", "")
            self.compute_char_brkpoints()
            self.man_segmented = None
            self.man_word_brkpoints = None

        if input_type == "man_segmented":
            self.man_segmented = input_line
            self.man_word_brkpoints = self.compute_word_brkpoints(self.man_segmented)
            self.unsegmented = input_line.replace("|", "")
            self.compute_char_brkpoints()
            self.compute_icu_segmented()
            self.icu_word_brkpoints = self.compute_word_brkpoints(self.icu_segmented)

        chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getRoot())
        chars_break_iterator.setText(self.unsegmented)
        self.char_brkpoints = [0]
        for brkpoint in chars_break_iterator:
            self.char_brkpoints.append(brkpoint)

    def compute_icu_segmented(self):
        """
        This function computes the ICU segmented version of the line.
        """
        words_break_iterator = BreakIterator.createWordInstance(Locale.getRoot())
        words_break_iterator.setText(self.unsegmented)
        self.icu_word_brkpoints = [0]
        for brkpoint in words_break_iterator:
            self.icu_word_brkpoints.append(brkpoint)
        self.icu_segmented = "|"
        for i in range(len(self.icu_word_brkpoints) - 1):
            self.icu_segmented += self.unsegmented[self.icu_word_brkpoints[i]: self.icu_word_brkpoints[i + 1]] + "|"

    def compute_word_brkpoints(self, input_line):
        """
        Given a segmented line, this function computes a list of word breakpoints. Note that it treats all "|" as a
        separator between two successive words.
        Args:
            input_line: the input segmented line.
        """
        word_brkpoints = []
        found_bars = 0
        for i in range(len(input_line)):
            if input_line[i] == '|':
                word_brkpoints.append(i - found_bars)
                found_bars += 1
        return word_brkpoints

    def compute_char_brkpoints(self):
        """
        This function uses ICU BreakIterator to identify and store extended grapheme clusters.
        """
        chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getRoot())
        chars_break_iterator.setText(self.unsegmented)
        self.char_brkpoints = [0]
        for brkpoint in chars_break_iterator:
            self.char_brkpoints.append(brkpoint)

    def display(self):
        """
        This function displays different versions of the line (unsegmented, icu_segmented, man_segmented) and also word
        break points for different types of segmenters and a list of extended grapheme clusters in the line.
        """
        print("unsegmented       : {}".format(self.unsegmented))
        print("icu_segmented     : {}".format(self.icu_segmented))
        print("man_segmented     : {}".format(self.man_segmented))
        print("icu_word_brkpoints: {}".format(self.icu_word_brkpoints))
        print("man_word_brkpoints: {}".format(self.man_word_brkpoints))
        print("char_brkpoints    : {}".format(self.char_brkpoints))


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
    """
    def __init__(self, input_n, input_t, input_graph_clust_dic, input_embedding_dim, input_hunits, input_dropout_rate,
                 input_output_dim, input_epochs, input_training_data, input_evaluating_data):
        self.n = input_n
        self.t = input_t
        if self.t % self.n != 0:
            print("Warning: t is not divided by n")
        self.batch_size = self.t // self.n  # number of batches used to train the model. It is defined as t // n
        self.graph_clust_dic = input_graph_clust_dic
        self.clusters_num = len(self.graph_clust_dic.keys()) + 1  # number of grapheme clusters in graph_clust_dic
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
            input_str = get_best_data_text(starting_text=1, ending_text=10, pseudo=False)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            if self.t > x_data.shape[0]:
                print("Warning: size of the training data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]

        elif self.training_data == "pseudo BEST":
            # this chunk of data has ~ 2*10^6 data points
            input_str = get_best_data_text(starting_text=1, ending_text=10, pseudo=True)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            if self.t > x_data.shape[0]:
                print("Warning: size of the training data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]

        elif self.training_data == "my":
            # this chunk of data has ~ 2*10^6 data points
            input_str = combine_lines_of_file("./Data/my_train.txt", input_type="unsegmented",
                                                  output_type="icu_segmented")
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
            input_str = get_best_data_text(starting_text=10, ending_text=20, pseudo=False)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            if self.t > x_data.shape[0]:
                print("Warning: size of the validation data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        elif self.training_data == "pseudo BEST":
            # this chunk of data has ~ 2*10^6 data points
            input_str = get_best_data_text(starting_text=10, ending_text=20, pseudo=True)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            if self.t > x_data.shape[0]:
                print("Warning: size of the validation data is less than self.t")
            x_data = x_data[:self.t]
            y_data = y_data[:self.t, :]
        elif self.training_data == "my":
            # this chunk of data has ~ 2*10^6 data points
            input_str = combine_lines_of_file("./Data/my_valid.txt", input_type="unsegmented",
                                                  output_type="icu_segmented")
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
            if self.t > x_data.shape[0]:
                print("Warning: size of the training data is less than self.t")
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
            input_str = get_best_data_text(starting_text=40, ending_text=45, pseudo=False)
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
        elif self.evaluating_data == "SAFT":
            input_str = combine_lines_of_file("./Data/SAFT/test.txt", input_type="man_segmented",
                                                  output_type="man_segmented")
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
        elif self.evaluating_data == "my":
            input_str = combine_lines_of_file("./Data/my_test.txt", input_type="unsegmented",
                                                  output_type="icu_segmented")
            x_data, y_data = get_trainable_data(input_str, self.graph_clust_dic)
        else:
            print("Warning: no implementation for this evaluation data exists!")
        test_batch_size = x_data.shape[0]//self.n
        test_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=test_batch_size,
                                             dim_output=self.output_dim)

        # Testing batch by batch (each batch of length self.n)
        all_test_input, all_actual_y = test_generator.generate_all_batches()
        all_y_hat = self.model.predict(all_test_input)
        test_acc_bies = []
        test_acc_f1 = []
        for i in range(test_batch_size):
            actual_y = all_actual_y[i, :, :]
            actual_y = get_bies_string_from_softmax(actual_y)
            y_hat = all_y_hat[i, :, :]
            y_hat = get_bies_string_from_softmax(y_hat)

            # Compute the BIES accuracy
            mismatch = diff_strings(actual_y, y_hat)
            test_acc_bies.append(1 - mismatch / len(actual_y))

            # Compute F1 score
            # Note: the actual_y cannot be used directly here, because it doesn't necessarily
            # give meaningful bies sequence. There for f1-score computed in this function is not quite precise
            y_hat_norm = normalize_bies(y_hat)
            test_acc_f1.append(compute_f1_score(true_bies=normalize_bies(actual_y), est_bies=y_hat_norm)[0])

        test_acc_bies = np.array(test_acc_bies)
        test_acc_f1 = np.array(test_acc_f1)
        print("the average BIES test accuracy in test_model function: {}".format(np.mean(test_acc_bies)))
        print("the average F1 test accuracy in test_model function: {}".format(np.mean(test_acc_f1)))
        # return np.mean(test_acc_bies)

    def test_text_line_by_line(self, file, line_limit):
        """
        This function tests the model fitted in self.train() using BEST data set. Unlike test_model() function, this
        function tests the model line by line. It combines very short lines together before testing.
        Args:
            file: the address of the file that is going to be tested
            line_limit: number of lines to be tested
        """
        test_acc_bies = []
        test_acc_f1 = []
        prev_str = ""
        line_counter = 0
        with open(file) as f:
            for line in f:
                if line_counter == line_limit:
                    break
                line = clean_line(line)
                if line == -1:
                    continue
                line_counter += 1
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

                # Use the manual predict function -- tf function doesn't always work properly for varying length strings
                y_hat = self.manual_predict(x_data)
                y_hat = get_bies_string_from_softmax(y_hat)
                actual_y = get_bies_string_from_softmax(y_data)

                # Compute the BIES accuracy
                mismatch = diff_strings(actual_y, y_hat)
                test_acc_bies.append(1 - mismatch / len(actual_y))

                # Compute F1 score
                y_hat_norm = normalize_bies(y_hat)
                test_acc_f1.append(compute_f1_score(true_bies=actual_y, est_bies=y_hat_norm)[0])

            print("the average BIES test accuracy (line by line) for file {} : {}".format(file, np.mean(test_acc_bies)))
            print("the average F1 test accuracy (line by line) for file {} : {}".format(file, np.mean(test_acc_f1)))
            return [test_acc_bies, test_acc_f1]

    def test_model_line_by_line(self):
        """
        This function uses the test_text_line_by_line() to test the model by a range of texts in BEST data set. The
        final score is the average of scores computed for each individual text.
        """
        all_bies_test_acc = []
        all_f1_test_acc = []
        if self.evaluating_data == "BEST":
            category = ["news", "encyclopedia", "article", "novel"]
            for text_num in range(40, 45):
                print("testing text {}".format(text_num))
                for cat in category:
                    text_num_str = "{}".format(text_num).zfill(5)
                    file = "./Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt"
                    [bies_test_acc, f1_test_acc] = self.test_text_line_by_line(file, line_limit=-1)
                    all_bies_test_acc += bies_test_acc
                    all_f1_test_acc += f1_test_acc
        elif self.evaluating_data == "my":
            file = "./Data/my_test_segmented.txt"
            num_lines = sum(1 for _line in open(file))
            line_limit = 2000
            if line_limit > num_lines:
                print("Warning: number of lines you are using is larger than the total numbe of lines in " + file)
            [bies_test_acc, f1_test_acc] = self.test_text_line_by_line(file, line_limit=line_limit)
            all_bies_test_acc += bies_test_acc
            all_f1_test_acc += f1_test_acc
        else:
            print("Warning: no implementation for this evaluation data exists!")
        print("the average BIES test accuracy by test_model_line_by_line function: {}".format(np.mean(all_bies_test_acc)))
        print("the average F1 test accuracy by test_model_line_by_line function: {}".format(np.mean(all_f1_test_acc)))

        # return np.mean(all_bies_test_acc)

    def manual_predict(self, test_input):
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
            input_graph_id = int(test_input[i])
            x_t = embedarr[input_graph_id, :]
            x_t = x_t.reshape(1, x_t.shape[0])
            h_fw, c_fw = compute_hc(weightLSTM, x_t, h_fw, c_fw)
            all_h_fw[i, :] = h_fw
        # print(all_h_fw)
        # print(all_h_fw.shape)

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

    def segment_arbitrary_line(self, input_line):
        """
        This function uses the lstm model to segmenent an unsegmented line. It is intended to be used to analyze errors.
        Args:
            input_line: the unsegmented input line
        """
        line = Line(input_line, "unsegmented")
        line_len = len(line.char_brkpoints) - 1
        x_data = np.zeros(shape=[line_len, 1])
        excess_grapheme_ids = max(self.graph_clust_dic.values()) + 1
        for i in range(line_len):
            char_start = line.char_brkpoints[i]
            char_finish = line.char_brkpoints[i + 1]
            curr_char = line.unsegmented[char_start: char_finish]
            x_data[i, 0] = self.graph_clust_dic.get(curr_char, excess_grapheme_ids)
        y_hat = word_segmenter.manual_predict(x_data)
        y_hat = get_bies_string_from_softmax(y_hat)
        y_hat = normalize_bies(y_hat)
        print("Input line     : {}".format(line.unsegmented))
        print("ICU segmented  : {}".format(line.icu_segmented))
        print("LSTM segmented : {}".format(get_segmented_string_from_bies(line, y_hat)))

    def get_model(self):
        return self.model

    def set_model(self, input_model):
        self.model = input_model


################################ Processing Thai ################################
# Adding space bars to the SAFT data around spaces
# add_additional_bars("./Data/SAFT/test_raw.txt", "./Data/SAFT/test.txt")

# Looking at the accuracy of the ICU on SAFT data set
# print("ICU BIES accuracy on SAFT data is {}.".format(compute_icu_accuracy("./Data/SAFT/test.txt")[0]))
# print("ICU F1 accuracy on SAFT data is {}.".format(compute_icu_accuracy("./Data/SAFT/test.txt")[1]))

# Preprocess the Thai language
# Thai_graph_clust_ratio, icu_bies_accuracy, icu_f1_accuracy = preprocess_thai(verbose=False)
# print("ICU BIES accuracy on BEST data is {}".format(icu_bies_accuracy))
# print("ICU F1 accuracy on BEST data is {}".format(icu_f1_accuracy))
# np.save(os.getcwd() + '/Data/Thai_graph_clust_ratio.npy', Thai_graph_clust_ratio)

# Loading the graph_clust from memory
graph_clust_ratio = np.load(os.getcwd() + '/Data/Thai_graph_clust_ratio.npy', allow_pickle=True).item()
# print(graph_clust_ratio)
# print_grapheme_clusters(ratios=graph_clust_ratio, thrsh=0.99)

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
'''
model_name = "Thai_temp"
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
                               input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=10, input_training_data="BEST", input_evaluating_data="BEST")

# Training and saving the model
word_segmenter.train_model()
word_segmenter.test_model()
# word_segmenter.test_model_line_by_line()
fitted_model = word_segmenter.get_model()
fitted_model.save("./Models/" + model_name)
np.save(os.getcwd() + "/Models/" + model_name + "/" + "weights", fitted_model.weights)
write_model_json(model_name, graph_clust_dic, fitted_model)
'''

# Choose one of the saved models to use
# '''
# Thai model 1: Bi-directional LSTM (trained on BEST), grid search
#     thrsh = 350, embedding_dim = 40, hunits = 40
# Thai model 2: Bi-directional LSTM (trained on BEST), grid search + manual reduction of hunits and embedding_size
#     thrsh = 350, embedding_dim = 20, hunits = 20
# Thai model 3: Bi-directional LSTM (trained on BEST), grid search + extreme manual reduction of hunits and embedding_size
#     thrsh = 350, embedding_dim = 15, hunits = 15
# Thai model 4: Bi-directional LSTM (trained on BEST), short BayesOpt choice for hunits and embedding_size
#     thrsh = 350, embedding_dim = 16, hunits = 23
# Thai model 5: Bi-directional LSTM (trained on BEST), A very parsimonious model
#     thrsh = 250, embedding_dim = 10, hunits = 10
# Thai temp: a temporary model, it should be used for storing new models

model_name = "Thai_model4_heavy"
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
# write_grapheme_clusters_dic_json(graph_clust_ratio, graph_thrsh)
word_segmenter = WordSegmenter(input_n=200, input_t=600000, input_graph_clust_dic=graph_clust_dic,
                               input_embedding_dim=input_embedding_dim, input_hunits=input_hunits,
                               input_dropout_rate=0.2, input_output_dim=4, input_epochs=15,
                               input_training_data="BEST", input_evaluating_data="BEST")
word_segmenter.set_model(model)

# Testing the model by arbitrary sentences
# line = "แม้จะกะเวลาเอาไว้แม่นยำว่ากว่าเขาจะมาถึงก็คงประมาณหกโมงเย็น"
# word_segmenter.segment_arbitrary_line(line)
# x = input()

# Testing the model using large texts
word_segmenter.test_model()
word_segmenter.test_model_line_by_line()
# '''

# Code for testing the bies normalizer function
# Testing the normalizer
# input_bies = "bbiies"
# print(normalize_bies(input_bies))


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



