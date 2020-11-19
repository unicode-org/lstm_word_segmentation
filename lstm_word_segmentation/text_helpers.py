import numpy as np
from line import Line
from accuracy import Accuracy
import constants
from helpers import is_ascii


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


def get_lines_of_text(file, type):
    out = []
    with open(file) as f:
        for file_line in f:
            file_line = clean_line(file_line)
            if file_line == -1:
                continue
            line = Line(file_line, type)
            out.append(line)
    return out


def compute_icu_accuracy(filename):
    """
    This function uses a dataset with segmented lines to compute the accuracy of icu word breakIterator
    Args:
        filename: The path of the file
    """
    accuracy = Accuracy()
    with open(filename) as f:
        for file_line in f:
            file_line = clean_line(file_line)
            if file_line == -1:
                continue
            line = Line(file_line, "man_segmented")
            true_bies = line.get_bies(segmentation_type="man")
            icu_bies = line.get_bies(segmentation_type="icu")

            # Computing the bies accuracy and F1 score using icu_bies_str and true_bies_str
            accuracy.update(true_bies=true_bies.str, est_bies=icu_bies.str)
        return accuracy.get_bies_accuracy(), accuracy.get_f1_score()


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

