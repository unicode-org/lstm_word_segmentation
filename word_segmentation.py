# import PyICU as pyicu
import numpy as np
import os
import icu
import datetime
from icu import UnicodeString, BreakIterator, Locale
import matplotlib.pyplot as plt

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

def get_clean_text(starting_text, ending_text):
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

def get_trainable_data(input_str, times, n, graph_clust_ids):
    x_data = np.zeros(shape=[times, 1])
    y_data = np.zeros(shape=[times, 4])
    # Finding word breakpoints
    word_brkpoints = []
    found_bars = 0
    for i in range(len(input_str)):
        if input_str[i] == '|':
            word_brkpoints.append(i - found_bars)
            found_bars += 1
    unsegmented_str = input_str.replace("|", "")

    # Finding grapheme cluster breakpoints
    chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getUS())
    chars_break_iterator.setText(unsegmented_str)
    char_brkpoints = [0]
    for brkpoint in chars_break_iterator:
        char_brkpoints.append(brkpoint)

    # Finding bies
    true_bies = get_bies(char_brkpoints, word_brkpoints)
    true_bies_str = get_bies_string(true_bies)

    # Printing for debugging
    # print("INPUT STR: {}".format(input_str[:200]))
    # print("UNSEG STR: {}".format(unsegmented_str[:200]))
    # print("BIES  STR: {}".format(true_bies_str[:200]))

    # Creating x_data and y_data
    excess_grapheme_ids = max(graph_clust_ids.values())
    for i in range(times):
        char_st = char_brkpoints[i]
        char_fn = char_brkpoints[i+1]
        curr_char = unsegmented_str[char_st: char_fn]
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

# Playing with an custom input string
'''
# The current algorithm segments the [refrigerator] as [cold] + [cabinet], and [night] as [middle] + [night]
# (there are multiple words for night in thai), and [sun] as [horoscope] + [week]:
# input_str = "ฉันซื้อตู้เย็นเมื่อวานนี้และฉันชอบมันมากวันกลางคืนดวงอาทิตย์ฉันเป็นหมอฟัน"
input_str = "ชนะรัฐไทยด้วย"

# The current algorithm segments [dentist] as [doctor] + [teeth]. It seems it fails to detect almost all compound words
# because it is optimized based on finding consecutive words -- it finds the simplest (shortest) words
# input_str = "ฉันเป็นหมอฟัน"
# input_str = "คนขับรถผ่านแยกนี้ไม่มากนัก"

print("input_str:\n{}".format(input_str))

# Here chars is in fact grapheme clusters, I just use characters because of the naming of the function
words_break_iterator = BreakIterator.createWordInstance(Locale.getUS())
chars_break_iterator = BreakIterator.createCharacterInstance(Locale.getUS())
words_break_iterator.setText(input_str)
chars_break_iterator.setText(input_str)
word_brkpoints = [0]
for brkpoint in words_break_iterator:
    word_brkpoints.append(brkpoint)
char_brkpoints = [0]
for brkpoint in chars_break_iterator:
    char_brkpoints.append(brkpoint)
char_segmented_str = get_segmented_string(input_str, char_brkpoints)
word_segmented_str = get_segmented_string(input_str, word_brkpoints)
print("Chars:\n{}".format(char_segmented_str))
print("Words:\n{}".format(word_segmented_str))
# print(word_brkpoints)
# x = input()
bies = get_bies(char_brkpoints, word_brkpoints)
print_bies(input_str, bies)
'''

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
grapheme_clusters_ids = dict()
for key in graph_clust_ratio.keys():
    if cnt < thrsh-1:
        grapheme_clusters_ids[key] = cnt
    if cnt == thrsh-1:
        break
    cnt += 1
# print(grapheme_clusters_ids)

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


# Building the LSTM model using the segmented data
# '''
num_texts = 5
train_texts_first = 1
valid_texts_first = 10
test_texts_first = 30
input_str = get_clean_text(starting_text=train_texts_first, ending_text=train_texts_first+num_texts)
print(len(input_str))
times = 100000  # Number of characters that we cover
n = 50         # length of each batches
x_data, y_data = get_trainable_data(input_str, times, n, grapheme_clusters_ids)
train_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

input_str = get_clean_text(starting_text=valid_texts_first, ending_text=valid_texts_first+num_texts)
print(len(input_str))
x_data, y_data = get_trainable_data(input_str, times, n, grapheme_clusters_ids)
valid_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

input_str = get_clean_text(starting_text=test_texts_first, ending_text=test_texts_first+num_texts)
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
                    epochs=15, validation_data=valid_generator.generate(), validation_steps=valid_generator.times//
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

# '''

# Building the LSTM model using the segmented data
'''
num_texts = 5
train_texts_first = 1
valid_texts_first = 10
test_texts_first = 30
input_str = get_clean_text(starting_text=train_texts_first, ending_text=train_texts_first+num_texts)
print(len(input_str))
times = 100000  # Number of characters that we cover
n = 50         # length of each batches
x_data, y_data = get_pseudo_trainable_data(input_str, times, n, grapheme_clusters_ids)
train_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

input_str = get_clean_text(starting_text=valid_texts_first, ending_text=valid_texts_first+num_texts)
print(len(input_str))
x_data, y_data = get_pseudo_trainable_data(input_str, times, n, grapheme_clusters_ids)
valid_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=4,
                                      times=times)

input_str = get_clean_text(starting_text=test_texts_first, ending_text=test_texts_first+num_texts)
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






