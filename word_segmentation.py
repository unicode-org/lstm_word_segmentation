# import PyICU as pyicu
import numpy as np
import icu
import datetime
from icu import UnicodeString, BreakIterator, Locale
import matplotlib.pyplot as plt

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

    # print(line)
    # print(st_delete)
    # print(fn_delete)

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
    # print(line)
    # print(new_line)
    # print(st_delete)
    # print(fn_delete)
    # x = input()
    return new_line

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


# str = "sahand farhoodi is sweet"
# str = str.replace("s", 't')
# print(str)
# x = input()

grapheme_clusters_dic = dict()
icu_mismatch = 0
icu_total_bies_lengths = 0
for text_num in range(1, 20):
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
        line = remove_tags(line, "<NE>", "</NE>")
        line = remove_tags(line, "<AB>", "</AB>")

        removable_strings = ["<NE>", "</NE>", "<AB>", "</AB>", "\n"]
        # removable_strings = ["\n"]
        for bad_str in removable_strings:
            line = line.replace(bad_str, "")
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

# Assessing the precision of icu algorithm
# print("The bies precision of icu algorithm is {}".format(1 - icu_mismatch/icu_total_bies_lengths))


# Analyzing grapheme clusters for embeddings
# '''
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
# '''