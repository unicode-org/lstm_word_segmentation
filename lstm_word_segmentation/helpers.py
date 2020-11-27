import numpy as np
from . import constants
from icu import Char, Script, UCharCategory


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


def normalize_string(in_str, allowed_scripts):
    """
    Normalizes in_str by replacing letters and digits in other scripts with
    exemplar values.

    Args:
        in_str: String to process
        allowed_scripts: List of script short names (like "Mymr") to preserve
    """
    # TODO: Consider checking ScriptExtensions here as well
    output = ""
    for ch in in_str:
        ch_script = Script.getScript(ch)
        ch_type = Char.charType(ch)
        ch_bucket = constants.CHAR_TYPE_TO_BUCKET[ch_type]
        ch_digit = Char.digit(ch)
        if ch_script.getShortName() in allowed_scripts:
            # ch is in an allowed script:
            # copy directly to the output
            output += ch
        elif ch_bucket == 1:
            # ch is a letter in a disallowed script:
            # normalize to the sample char for that script
            output += Script.getSampleString(ch_script)
        elif ch_bucket == 3 and ch_digit != -1:
            # ch is a decimal digit in a disallowed script:
            # normalize to the zero digit in that numbering system
            output += chr(ord(ch) - ch_digit)
        elif ch_type == UCharCategory.CURRENCY_SYMBOL:
            # ch is a currency symbol in a disallowed script:
            # normalize to $
            output += "$"
        else:
            # all other characters:
            # copy directly to the output
            output += ch
    return output