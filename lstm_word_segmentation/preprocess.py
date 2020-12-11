from pathlib import Path
import numpy as np
from collections import Counter
from .text_helpers import get_lines_of_text, compute_accuracy, compute_accuracy_best
from . import constants


def evaluate_existing_algorithms(algorithm, data):
    """
    This function evaluates the algorithms that are imported, such as ICU and Deepcut for Thai
    Args:
        algorithm: the algortihm to be tested. It can be ICU or Deepcut for now
        data: the data to be used for testing algorithms. The values that it can take are:
    """
    acc = None
    if algorithm == "ICU":
        if data == "SAFT Thai":
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT/test.txt')
            acc = compute_accuracy(file, "icu")
        elif data == "BEST":
            acc = compute_accuracy_best(starting_text=40, ending_text=60, algorithm="icu", exclusive=False)
        elif data == "exclusive BEST":
            acc = compute_accuracy_best(starting_text=40, ending_text=60, algorithm="icu", exclusive=True)
        elif data == "SAFT Burmese":
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT_burmese_test_limited.txt')
            acc = compute_accuracy(file, "icu")
        elif data == "my":
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_test_segmented.txt')
            acc = compute_accuracy(file, "icu")

    if algorithm == "Deepcut":
        if data == "SAFT Thai":
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT/test.txt')
            acc = compute_accuracy(file, "icu")
        elif data == "BEST":
            acc = compute_accuracy_best(starting_text=40, ending_text=60, algorithm="deep", exclusive=False)
        elif data == "exclusive BEST":
            acc = compute_accuracy_best(starting_text=40, ending_text=60, algorithm="deep", exclusive=True)

    if acc is None:
        print("Warning: the evaluation for this combination of data and algorithm is not supported.")
    else:
        print(
            "{} accuracy on {} data set: BIES accuracy = {}, F1-score = {}".format(algorithm, data,
                                                                                   acc.get_bies_accuracy(),
                                                                                   acc.get_f1_score()))
    return acc


def find_grapheme_clusters(language, exclusive, verbose):
    """
    This function uses the BEST data set to
        1) compute the grapheme cluster dictionary that holds the frequency of different grapheme clusters
    Args:
        language: the language that we want to used to find grapheme clusters. It can be "Thai" or "Burmese"
        verbose: shows if we want to see how the algorithm is working or not
        exclusive: determines to use a data set where all code points are in the script associated with the language or
                   not
    """
    grapheme_clusters_dic = Counter()
    lines = []

    # For Thai use BEST data set
    if language == "Thai":
        num_texts = 96
        for text_num in range(1, num_texts):
            for cat in ["news", "encyclopedia", "article", "novel"]:
                text_num_str = "{}".format(text_num).zfill(5)
                if exclusive:
                    file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/exclusive_Best/{}/{}_".
                                         format(cat, cat) + text_num_str + ".txt")
                else:
                    file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/Best/{}/{}_".format(cat, cat) +
                                         text_num_str + ".txt")

                lines += get_lines_of_text(file=file, type_of_lines="man_segmented")

    # For Burmese use "my" data set
    elif language == "Burmese":
        if exclusive:
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/my.txt")
        else:
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/my.txt")  # use appropriate file here
        lines = get_lines_of_text(file=file, type_of_lines="man_segmented")

    else:
        print("Warning: the input language is not supported")

    for line in lines:
        grapheme_clusters_dic += line.get_grapheme_clusters()
        if verbose:
            line.display()

    # Saving the dictionary based on frequency of grapheme clusters
    graph_clust_freq = dict(grapheme_clusters_dic)
    graph_clust_freq = {k: v for k, v in sorted(graph_clust_freq.items(), key=lambda item: item[1], reverse=True)}
    graph_clust_ratio = graph_clust_freq
    total = sum(graph_clust_ratio.values(), 0.0)
    graph_clust_ratio = {k: v / total for k, v in graph_clust_ratio.items()}

    # Saving the dictionary
    save_file = ""
    if language == "Thai":
        save_file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/Thai_graph_clust_ratio.npy")
        if exclusive:
            save_file = Path.joinpath(Path(__file__).parent.parent.absolute(),
                                      "Data/Thai_exclusive_graph_clust_ratio.npy")
    if language == "Burmese":
        save_file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/Burmese_graph_clust_ratio.npy")
        if exclusive:
            save_file = Path.joinpath(Path(__file__).parent.parent.absolute(),
                                      "Data/Burmese_exclusive_graph_clust_ratio.npy")
    np.save(str(save_file), graph_clust_ratio)


def make_thai_burmese_dictionary():
    new_dic = Counter()
    for ch in constants.THAI_GRAPH_CLUST_RATIO.keys():
        new_dic[ch] += constants.THAI_GRAPH_CLUST_RATIO[ch]
    for ch in constants.BURMESE_GRAPH_CLUST_RATIO.keys():
        new_dic[ch] += constants.BURMESE_GRAPH_CLUST_RATIO[ch]
    new_dic = dict(new_dic)
    new_dic = {k: v for k, v in sorted(new_dic.items(), key=lambda item: item[1], reverse=True)}
    total = sum(new_dic.values(), 0.0)
    graph_clust_ratio = {k: v / total for k, v in new_dic.items()}
    save_file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/Thai_Burmese_graph_clust_ratio.npy")
    np.save(str(save_file), graph_clust_ratio)
