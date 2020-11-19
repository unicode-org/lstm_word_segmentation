import numpy as np
from line import Line
from accuracy import Accuracy
from collections import Counter
from text_helpers import get_lines_of_text, compute_icu_accuracy, add_additional_bars


def preprocess_saft_data():
    add_additional_bars("./Data/SAFT/test_raw.txt", "./Data/SAFT/test.txt")
    print("ICU BIES accuracy on SAFT data is {}".format(compute_icu_accuracy("./Data/SAFT/test.txt")[0]))
    print("ICU F1 accuracy on SAFT data is {}".format(compute_icu_accuracy("./Data/SAFT/test.txt")[1]))


def preprocess_thai(verbose):
    """
    This function uses the BEST data set to
        1) compute the grapheme cluster dictionary that holds the frequency of different grapheme clusters
        2) demonstrate the performance of icu word breakIterator and compute its accuracy
    Args:
        verbose: shows if we want to see how the algorithm is working or not
    """
    grapheme_clusters_dic = Counter()
    accuracy = Accuracy()
    for cat in ["news", "encyclopedia", "article", "novel"]:
        for text_num in range(1, 96):
            text_num_str = "{}".format(text_num).zfill(5)
            file = "./Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt"
            lines = get_lines_of_text(file=file, type="man_segmented")

            for line in lines:
                # Storing the grapheme clusters and their frequency in the dictionary
                grapheme_clusters_dic += line.get_grapheme_clusters()

                # Computing BIES corresponding to the manually segmented and ICU segmented
                true_bies = line.get_bies(segmentation_type="man")
                icu_bies = line.get_bies(segmentation_type="icu")

                # Computing the bies accuracy and F1 score using icu_bies_str and true_bies_str
                accuracy.update(true_bies=true_bies.str, est_bies=icu_bies.str)

                # Demonstrate how icu segmenter works
                if verbose:
                    line.display()

    graph_clust_freq = dict(grapheme_clusters_dic)
    graph_clust_freq = {k: v for k, v in sorted(graph_clust_freq.items(), key=lambda item: item[1], reverse=True)}
    graph_clust_ratio = graph_clust_freq
    total = sum(graph_clust_ratio.values(), 0.0)
    graph_clust_ratio = {k: v / total for k, v in graph_clust_ratio.items()}

    print("ICU BIES accuracy on BEST data is {}".format(accuracy.get_bies_accuracy()))
    print("ICU F1 accuracy on BEST data is {}".format(accuracy.get_f1_score()))
    np.save(os.getcwd() + '/Data/Thai_graph_clust_ratio.npy', graph_clust_ratio)


def preprocess_burmese(verbose):
    """
    This function uses the Google corpus crawler Burmese data set to
        1) Clean it by removing tabs from start of it
        1) Compute the grapheme cluster dictionary that holds the frequency of different grapheme clusters
        2) Demonstrate the performance of icu word breakIterator and compute its accuracy
    Args:
        verbose: shows if we want to see how the algorithm is working or not
    """
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


