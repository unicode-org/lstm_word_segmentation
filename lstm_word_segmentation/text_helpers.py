import numpy as np
from pathlib import Path
from .line import Line
from .accuracy import Accuracy
from .helpers import is_ascii
from icu import Char, Script, UCharCategory
from . import constants

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


def add_additional_bars(read_filename, write_filename):
    """
    This function reads a segmented file and add bars around each space in it if appropriate bars don't exist. It
    assumes that spaces are used as breakpoints in the segmentation (just like "|")
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


def permute_lines_of_text(filename, permutated_filename):
    """
    This function first divides line of an input file into buckets where each bucket is a fixed number of successive
    lines in the file, and then write down these buckets of lines into a new file randomely (without repetition).
    Args:
        filename: the input file
        permutated_filename: the output file
    """
    num_lines = sum(1 for _line in open(filename))
    bucket_size = 20
    permutated_buckets = np.random.permutation(num_lines//bucket_size)
    new_file = open(permutated_filename, 'w')
    lines_list = [line for line in open(filename)]
    for bucket_id in permutated_buckets:
        for line in lines_list[bucket_id: bucket_id+bucket_size]:
            new_file.write(line)
    new_num_lines = sum(1 for _line in open(permutated_filename))


def divide_train_test_data(input_text, train_text, valid_text, test_text, line_limit):
    """
    This function divides a file into three new files, that contain train data, validation data, and test data
    It first divide lines of the text into small buckets (e.g. 20 lines per each bucket) and then distribute each
    bucket between train, validation and test texts. This way, we will in each of the resulting texts, we have samples
    from all ovor the original text.
    Args:
        input_text: address of the original file
        train_text: address to store the train data in it
        valid_text: address to store the validation data in it
        test_text: address to store the test file in it
        line_limit: the number of lines that we want to process
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
            if line_counter > line_limit:
                break
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


def get_lines_of_text(file, type_of_lines):
    """
    Given a file, this function returns a list of line objects in that file.
    Args:
        file: Address of the file
        type_of_lines: It shows what is the type of sentences in the file. It can take 3 different values: unsegmented,
        man_segmented, or icu_segmented
    """
    out = []
    with open(file) as f:
        for file_line in f:
            file_line = clean_line(file_line)
            if file_line == -1:
                continue
            # If the line is unsegmented, we removing the bars from start and end of the clean line
            if type_of_lines == "unsegmented":
                file_line = file_line[1:-2]
            line = Line(file_line, type_of_lines)
            out.append(line)
    return out


def get_whole_file_segmented(filename, input_type, output_type):
    """
    This function returns a single segmented line that contains all lines in a text file. If the output is supposed to
    be manually segmented, it just combines the manually segmented sentences inside the file. If the output is supposed
    to be icu segmented, it first combines all unsegmented versions of line of the text into a single line, and then use
    icu to segment that single line. This decision is based on the fact that some code points in some Burmese lines are
    not valid code points (see line 457457 of the my_train.txt).
    Args:
        filename: address of the input file
        input_type: determines if the input is unsegmented, manually segmented, or ICU segmented
        output_type: determines if the output is manually segmented or ICU segmented
    """
    lines = get_lines_of_text(filename, input_type)
    all_file_line = ""
    if output_type == "man_segmented":
        for i in range(len(lines)):
            all_file_line += lines[i].man_segmented
            if i != len(lines) - 1:
                all_file_line += " "
        return all_file_line
    if output_type == "icu_segmented":
        for i in range(len(lines)):
            all_file_line += lines[i].unsegmented
            if i != len(lines)-1:
                all_file_line += " "
        all_file_line = Line(all_file_line, "unsegmented")
        return all_file_line.icu_segmented


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
        pseudo: if True, it means we use pseudo segmented data, if False, we use BEST manually segmentation
    """
    out_str = ""
    category = ["news", "encyclopedia", "article", "novel"]
    for text_num in range(starting_text, ending_text):
        for cat in category:
            text_num_str = "{}".format(text_num).zfill(5)
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/Best/{}/{}_".format(cat, cat) +
                                 text_num_str + ".txt")
            output_type = "man_segmented"
            if pseudo:
                output_type = "icu_segmented"
            new_line = get_whole_file_segmented(filename=file, input_type="man_segmented", output_type=output_type)
            if len(out_str) == 0:
                out_str = new_line
            else:
                out_str = out_str + " " + new_line
    return out_str


def compute_accuracy(file, segmentation_type):
    """
    This function uses a file with manually segmented lines to compute the accuracy of icu word breakIterator
    Args:
        file: The file to be used for computing accuracy
        segmentation_type: Indicates what algorithm we want to test. For now, it can be "icu" or "deep".
    """
    accuracy = Accuracy()
    lines = get_lines_of_text(file, "man_segmented")
    for line in lines:
        true_bies = line.get_bies(segmentation_type="man")
        algo_bies = line.get_bies(segmentation_type=segmentation_type)
        # print(line.unsegmented)
        # print(line.icu_segmented)
        # print(line.man_segmented)
        # print(true_bies.str)
        # print(algo_bies.str)
        # x = input()
        accuracy.update(true_bies=true_bies.str, est_bies=algo_bies.str)
    return accuracy


def compute_accuracy_best(starting_text, ending_text, algorithm):
    """
    This funciton computes accuracy of an algorithm on Best data set.
    Args:
        starting_text: number or the smallest text
        ending_text: number or the largest text + 1
        algorithm: the algorithm to be tested. It can be "icu" or "deep" for now.
    """
    category = ["news", "encyclopedia", "article", "novel"]
    accuracy = Accuracy()
    for text_num in range(starting_text, ending_text):
        print(text_num)
        for cat in category:
            text_num_str = "{}".format(text_num).zfill(5)
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/Best/{}/{}_".format(cat, cat) +
                                 text_num_str + ".txt")
            accuracy.merge_accuracy(compute_accuracy(file, segmentation_type=algorithm))
    return accuracy


def break_lines_based_on_spaces(input_texts, output_text):
    """
    This function uses every space in the text to form a new line.
    Args:
        input_texts: list of address of a group of the input files
        output_text: address of the new file
    """
    output_file = open(output_text, 'w')
    for input_text in input_texts:
        print(input_text)
        with open(input_text) as f:
            for line in f:
                line = clean_line(line)
                if line == -1:
                    continue
                new_lines = line.split(" ")
                for new_line in new_lines:
                    output_file.write(new_line + "\n")


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


def merge_two_texts(input_texts1, input_texts2, output_text, line_limit):
    """
    This function merges two texts into one new text. Every line of text 1 appears before every line of text 2.
    Args:
        input_texts1: list of address of the first group of the input files
        input_texts2: list of address of the second group of the input files
        output_text: address of the resulting file
        line_limit: number of lines to be included from each group of input texts
    """
    output_file = open(output_text, 'w')
    line_counter = 0
    for input_text in input_texts1:
        if line_counter >= line_limit:
            break
        with open(input_text) as f:
            for line in f:
                if line_counter >= line_limit:
                    break
                output_file.write(line)
                line_counter += 1

    line_counter = 0
    for input_text in input_texts2:
        if line_counter >= line_limit:
            break
        with open(input_text) as f:
            for line in f:
                if line_counter >= line_limit:
                    break
                output_file.write(line)
                line_counter += 1




