from pathlib import Path
from lstm_word_segmentation.word_segmenter import pick_lstm_model
from lstm_word_segmentation.text_helpers import get_lines_of_text
from lstm_word_segmentation.line import Line
from lstm_word_segmentation.preprocess import evaluate_existing_algorithms
import deepcut
import timeit


# Analyzing how different word segmenters perform for a set of unsegmented lines
# '''
# Picking models for error analysis
word_segmenter1 = pick_lstm_model(model_name="Thai_graphclust_exclusive_model4_heavy", embedding="grapheme_clusters_tf",
                                  train_data="BEST", eval_data="BEST")

word_segmenter2 = pick_lstm_model(model_name="Thai_graphclust_model5_heavy", embedding="grapheme_clusters_tf",
                                  train_data="BEST", eval_data="BEST")

word_segmenter3 = pick_lstm_model(model_name="Thai_graphclust_model7_heavy", embedding="grapheme_clusters_tf",
                                  train_data="BEST", eval_data="BEST")

word_segmenter4 = pick_lstm_model(model_name="Thai_graphclust_exclusive_model4_heavy", embedding="grapheme_clusters_tf",
                                  train_data="exclusive BEST", eval_data="exclusive BEST")

word_segmenter5 = pick_lstm_model(model_name="Thai_codepoints_exclusive_model4_heavy", embedding="codepoints",
                                  train_data="exclusive BEST", eval_data="exclusive BEST")

word_segmenter6 = pick_lstm_model(model_name="Thai_codepoints_exclusive_model7_heavy", embedding="codepoints",
                                  train_data="exclusive BEST", eval_data="exclusive BEST")

word_segmenters = [word_segmenter1, word_segmenter2, word_segmenter3, word_segmenter4, word_segmenter5, word_segmenter6]

# Show output of models
verbose = True

# Write the output of models into a output_file defined below
write = False
if write:
    output_file = Path.joinpath(Path(__file__).parent.absolute(), 'Data/wiki_thai_sample_exclusive_results.txt')
    output = open(str(output_file), 'w')

# Read the input from a file directly. The alternative is inserting them directly below
read = False
if read:
    file = Path.joinpath(Path(__file__).parent.absolute(), 'Data/wiki_thai_sample_exclusive.txt')
    lines = get_lines_of_text(file, "unsegmented")
else:
    inp_lines = ["|เพราะ|เขา|เห็น|โอกาส|ใน|การ|ซื้อ|", "|การ|เดินทาง|ใน|", "|นั่ง|นายก|ฯ|ต่อ|สมัย|หน้า|",
                 "|พร้อม|จัดตั้ง|",
                 "|เพราะ|ดนตรี|ที่|ชอบ|นั้น|"]
    # inp_lines = ["|ဖော်ပြ|ထားသည့်|", "|အသားအရောင်|အားဖြင့်|", "|သဘာဝ|အားဖြင့်|", "|ထို့ပြင်|", "|နိုင်ငံရေး|ဆိုင်ရာ|"]
    lines = []
    for inp_line in inp_lines:
        lines.append(Line(inp_line, "man_segmented"))

# Running different models
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

# Measuring the time that different models need to evaluate texts
# '''
word_segmenter1 = pick_lstm_model(model_name="Thai_codepoints_exclusive_model4_heavy", embedding="codepoints",
                                  train_data="exclusive BEST", eval_data="exclusive BEST")

word_segmenter2 = pick_lstm_model(model_name="Thai_graphclust_model5_heavy", embedding="grapheme_clusters_tf",
                                  train_data="BEST", eval_data="BEST")

word_segmenter3 = pick_lstm_model(model_name="Thai_graphclust_model7_heavy", embedding="grapheme_clusters_tf",
                                  train_data="BEST", eval_data="BEST")

word_segmenter4 = pick_lstm_model(model_name="Thai_codepoints_exclusive_model4_heavy", embedding="codepoints",
                                  train_data="exclusive BEST", eval_data="exclusive BEST")

word_segmenter5 = pick_lstm_model(model_name="Thai_codepoints_exclusive_model5_heavy", embedding="codepoints",
                                  train_data="exclusive BEST", eval_data="exclusive BEST")

word_segmenter6 = pick_lstm_model(model_name="Thai_codepoints_exclusive_model7_heavy", embedding="codepoints",
                                  train_data="exclusive BEST", eval_data="exclusive BEST")

word_segmenters = [word_segmenter1, word_segmenter2, word_segmenter3, word_segmenter4, word_segmenter5, word_segmenter6]

# Testing all models using the same BEST data: texts 40-60
for word_seg in word_segmenters:
    start = timeit.default_timer()
    word_seg.test_model_line_by_line(verbose=False, fast=True)
    stop = timeit.default_timer()
    print("{} Time using test_line_by_line: {}".format(word_seg.name, stop - start))

# ICU
start = timeit.default_timer()
evaluate_existing_algorithms(algorithm="ICU", data="BEST", fast=True)
stop = timeit.default_timer()
print("ICU Time: {}".format(stop-start))

# Deepcut
start = timeit.default_timer()
evaluate_existing_algorithms(algorithm="Deepcut", data="BEST", fast=True)
stop = timeit.default_timer()
print("Deepcut Time: {}".format(stop-start))
# '''
