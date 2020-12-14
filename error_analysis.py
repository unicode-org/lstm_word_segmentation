from pathlib import Path
from lstm_word_segmentation.word_segmenter import pick_lstm_model
from lstm_word_segmentation.text_helpers import get_lines_of_text
from lstm_word_segmentation.line import Line
import deepcut
import timeit


# Picking models for error analysis
word_segmenter1 = pick_lstm_model(model_name="Burmese_graphclust_model4_heavy", embedding="grapheme_clusters_tf",
                                  train_data="my", eval_data="my")
word_segmenter2 = pick_lstm_model(model_name="Burmese_genvec_1235", embedding="generalized_vectors_1235",
                                  train_data="my", eval_data="my")

# word_segmenter1 = pick_lstm_model(model_name="Thai_graphclust_model4_heavy", embedding="grapheme_clusters_tf",
#                                   train_data="BEST", eval_data="BEST")
#
# word_segmenter2 = pick_lstm_model(model_name="Thai_graphclust_model5_heavy", embedding="grapheme_clusters_tf",
#                                   train_data="BEST", eval_data="BEST")
#
# word_segmenter3 = pick_lstm_model(model_name="Thai_graphclust_model7_heavy", embedding="grapheme_clusters_tf",
#                                   train_data="BEST", eval_data="BEST")

# word_segmenter4 = pick_lstm_model(model_name="Thai_graphclust_exclusive_model4_heavy", embedding="grapheme_clusters_tf",
#                                   train_data="exclusive BEST", eval_data="exclusive BEST")
#
# word_segmenter5 = pick_lstm_model(model_name="Thai_codepoints_exclusive_model4_heavy", embedding="codepoints",
#                                   train_data="exclusive BEST", eval_data="exclusive BEST")
#
# word_segmenter6 = pick_lstm_model(model_name="Thai_codepoints_exclusive_model7_heavy", embedding="codepoints",
#                                   train_data="exclusive BEST", eval_data="exclusive BEST")

# Testing the model by arbitrary sentences
# '''
verbose = True
write = False
# Use lines in a given file
file = Path.joinpath(Path(__file__).parent.absolute(), 'Data/wiki_thai_sample_exclusive.txt')
lines = get_lines_of_text(file, "unsegmented")
output_file = Path.joinpath(Path(__file__).parent.absolute(), 'Data/wiki_thai_sample_exclusive_results.txt')
output = open(str(output_file), 'w')

# Use the following list directly
# inp_lines = ["|เพราะ|เขา|เห็น|โอกาส|ใน|การ|ซื้อ|", "|การ|เดินทาง|ใน|", "|นั่ง|นายก|ฯ|ต่อ|สมัย|หน้า|", "|พร้อม|จัดตั้ง|",
#              "|เพราะ|ดนตรี|ที่|ชอบ|นั้น|"]
inp_lines = ["|ဖော်ပြ|ထားသည့်|", "|အသားအရောင်|အားဖြင့်|", "|သဘာဝ|အားဖြင့်|", "|ထို့ပြင်|", "|နိုင်ငံရေး|ဆိုင်ရာ|"]
lines = []
for inp_line in inp_lines:
    lines.append(Line(inp_line, "man_segmented"))

# word_segmenters = [word_segmenter1, word_segmenter2, word_segmenter3, word_segmenter4, word_segmenter5, word_segmenter6]
word_segmenters = [word_segmenter1, word_segmenter2]
for line in lines:
    deepcut_words = deepcut.tokenize(line.unsegmented)
    deepcut_segmented = "|"
    for word in deepcut_words:
        deepcut_segmented += word + "|"
    if verbose:
        print("***************************************************************************************************")
        print("{:<40} : {}".format("Unsegmented", line.unsegmented))
        print("{:<40} : {}".format("Deepcut", deepcut_segmented))
        print("{:<40} : {}".format("ICU", line.icu_segmented))
        for word_seg in word_segmenters:
            print("{:<40} : {}".format(word_seg.name, word_seg.segment_arbitrary_line(line.unsegmented)))
        print("***************************************************************************************************")
    if write:
        output.write("********************************************************************************\n")
        output.write("{:<40} : {}\n".format("Unsegmented", line.unsegmented))
        output.write("{:<40} : {}\n".format("Deepcut", deepcut_segmented))
        output.write("{:<40} : {}\n".format("ICU", line.icu_segmented))
        for word_seg in word_segmenters:
            output.write("{:<40} : {}\n".format(word_seg.name, word_seg.segment_arbitrary_line(line.unsegmented)))
        output.write("********************************************************************************\n")
# '''

# Measuring the time that a model needs to evaluate texts
'''
word_seg = word_segmenter6
start = timeit.default_timer()
word_seg.test_model_line_by_line()
stop = timeit.default_timer()
print("{} Time: {}".format(word_seg.name, stop-start))
'''