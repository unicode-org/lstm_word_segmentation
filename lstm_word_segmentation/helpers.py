import numpy as np
from . import constants


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


def print_grapheme_clusters(thrsh, language):
    """
    This function analyzes the grapheme clusters, to see what percentage of them form which percent of the text, and
    provides a histogram that shows frequency of grapheme clusters
    ratios: a dictionary that holds the ratio of text that is represented by each grapheme cluster
    Args:
        thrsh: shows what percent of the text we want to be covered by grapheme clusters
        language: shows the language that we are working with
    """
    ratios = None
    if language == "Thai":
        ratios = constants.THAI_GRAPH_CLUST_RATIO
    if language == "Burmese":
        ratios = constants.BURMESE_GRAPH_CLUST_RATIO
    if ratios is None:
        print("No grapheme cluster dictionary has been computed for the input language")
        return
    cum_sum = 0
    cnt = 0
    for val in ratios.values():
        cum_sum += val
        cnt += 1
        if cum_sum > thrsh:
            break
    print("number of different grapheme clusters in {} = {}".format(language, len(ratios.keys())))
    print("{} grapheme clusters form {} of the text".format(cnt, thrsh))
