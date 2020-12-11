from pathlib import Path
from lstm_word_segmentation.word_segmenter import pick_lstm_model
from lstm_word_segmentation.text_helpers import get_lines_of_text
from lstm_word_segmentation.line import Line
import deepcut
import timeit


# Picking models for error analysis
word_segmenter1 = pick_lstm_model(model_name="Thai_model4", embedding="grapheme_clusters_tf", train_data="BEST",
                                  eval_data="BEST")

word_segmenter2 = pick_lstm_model(model_name="Thai_exclusive_model4", embedding="grapheme_clusters_tf",
                                  train_data="exclusiveBEST", eval_data="exclusive BEST")

word_segmenter3 = pick_lstm_model(model_name="Thai_model4_heavy", embedding="grapheme_clusters_tf", train_data="BEST",
                                  eval_data="BEST")

word_segmenter4 = pick_lstm_model(model_name="Thai_exclusive_model4_heavy", embedding="grapheme_clusters_tf",
                                  train_data="exclusive BEST", eval_data="exclusive BEST")

word_segmenter5 = pick_lstm_model(model_name="Thai_genvec_123", embedding="generalized_vectors_123", train_data="BEST",
                                  eval_data="BEST")

word_segmenter6 = pick_lstm_model(model_name="Thai_codepoints_model4_heavy", embedding="codepoints",
                                  train_data="exclusive BEST", eval_data="exclusive BEST")


# Testing the model by arbitrary sentences
# '''
verbose = True
# Use lines in a given file
file = Path.joinpath(Path(__file__).parent.absolute(), 'Data/wiki_thai_sample_exclusive.txt')
lines = get_lines_of_text(file, "unsegmented")
output_file = Path.joinpath(Path(__file__).parent.absolute(), 'Data/wiki_thai_sample_exclusive_results.txt')
output = open(str(output_file), 'w')

# Use the following list directly
# inp_lines = ["|เพราะ|เขา|เห็น|โอกาส|ใน|การ|ซื้อ|", "|การ|เดินทาง|ใน|", "|นั่ง|นายก|ฯ|ต่อ|สมัย|หน้า|", "|พร้อม|จัดตั้ง|",
#              "|เพราะ|ดนตรี|ที่|ชอบ|นั้น|"]
# lines = []
# for inp_line in inp_lines:
#     lines.append(Line(inp_line, "man_segmented"))

for line in lines:
    deepcut_words = deepcut.tokenize(line.unsegmented)
    deepcut_segmented = "|"
    for word in deepcut_words:
        deepcut_segmented += word + "|"
    if verbose:
        print("***************************************************************************************************")
        print("Unsegmented                   : {}".format(line.unsegmented))
        print("Deepcut                       : {}".format(deepcut_segmented))
        print("ICU                           : {}".format(line.icu_segmented))
        print("Thai_model4                   : {}".format(word_segmenter1.segment_arbitrary_line(line.unsegmented)))
        print("Thai_exclusive_model4         : {}".format(word_segmenter2.segment_arbitrary_line(line.unsegmented)))
        print("Thai_model4_heavy             : {}".format(word_segmenter3.segment_arbitrary_line(line.unsegmented)))
        print("Thai_exclusive_model4_heavy   : {}".format(word_segmenter4.segment_arbitrary_line(line.unsegmented)))
        print("Thai_genvec123                : {}".format(word_segmenter5.segment_arbitrary_line(line.unsegmented)))
        print("Thai_codepoints_mpodel4_heavy : {}".format(word_segmenter6.segment_arbitrary_line(line.unsegmented)))
        print("***************************************************************************************************")

    output.write("********************************************************************************\n")
    output.write("Unsegmented                        : {}\n".format(line.unsegmented))
    output.write("Deepcut                            : {}\n".format(deepcut_segmented))
    output.write("ICU                                : {}\n".format(line.icu_segmented))
    # output.write("Thai_model4                 : {}\n".format(word_segmenter1.segment_arbitrary_line(line.unsegmented)))
    # output.write("Thai_exclusive_model4       : {}\n".format(word_segmenter2.segment_arbitrary_line(line.unsegmented)))
    # output.write("Thai_model4_heavy           : {}\n".format(word_segmenter3.segment_arbitrary_line(line.unsegmented)))
    # output.write("Thai_exclusive_model4_heavy : {}\n".format(word_segmenter4.segment_arbitrary_line(line.unsegmented)))
    # output.write("Thai_genvec123              : {}\n".format(word_segmenter5.segment_arbitrary_line(line.unsegmented)))
    # output.write("Thai_codepoints             : {}\n".format(word_segmenter6.segment_arbitrary_line(line.unsegmented)))
    output.write("LSTM Grapheme Clusters Embedding   : {}\n".format(word_segmenter4.segment_arbitrary_line(line.unsegmented)))
    output.write("LSTM Generalized Vectors Embedding : {}\n".format(word_segmenter5.segment_arbitrary_line(line.unsegmented)))
    output.write("LSTM Code points Embedding         : {}\n".format(word_segmenter6.segment_arbitrary_line(line.unsegmented)))
    output.write("********************************************************************************\n")
# '''

# Measuring the time that each model need to evaluate texts
'''
start = timeit.default_timer()
word_segmenter6.test_model_line_by_line()
stop = timeit.default_timer()
print("{} Time: {}".format(word_segmenter6.name, stop-start))
'''