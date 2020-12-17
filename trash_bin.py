def normalize_bies(bies_str, bies_probs):
    """
    This function normalizes the input bies string to generate a bies string that makes sense. For example the output
    won't have substring such as "biiis", "biese" or "siie"
    Args:
        bies_str: The input bies string
    """
    if len(bies_str) == 0:
        return 's'
    out_bies = ""
    start_of_word = True
    i = 0

    while i < len(bies_str):
        if start_of_word:
            if i == len(bies_str) - 1 or bies_str[i] == 's':
                out_bies += 's'
                start_of_word = True
                i += 1
                continue
            elif bies_str[i] == 'b':
                out_bies += bies_str[i]
                start_of_word = False
                i += 1
                continue
            elif bies_str[i] in ['i', 'e']:
                if bies_str[i+1] in ['i', 'e']:
                    out_bies += 'b'
                    start_of_word = False
                    i += 1
                    continue
                elif bies_str[i+1] == 'b':
                    # Decide between "bi" and "sb":
                    if bies_probs[i, 0] * bies_probs[i+1, 1] > bies_probs[i, 3] * bies_probs[i+1, 0]:
                        out_bies += "bi"
                    else:
                        out_bies += "sb"
                    start_of_word = False
                    i += 2
                    continue
                elif bies_str[i+1] == 's':
                    # Decide between "be" and "ss":
                    if bies_probs[i, 0] * bies_probs[i+1, 2] > bies_probs[i, 3] * bies_probs[i+1, 3]:
                        out_bies += "be"
                    else:
                        out_bies += "ss"
                    start_of_word = True
                    i += 2
                    continue
        if not start_of_word:
            if bies_str[i] == 'i':
                if i == len(bies_str) - 1:
                    out_bies += 'e'
                    start_of_word = True
                    i += 1
                    continue
                elif bies_str[i+1] in ['i', 'e']:
                    out_bies += bies_str[i]
                    i += 1
                    continue
                elif bies_str[i+1] == 'b':
                    # Decide between "eb" and "ii":
                    if bies_probs[i, 2] * bies_probs[i+1, 0] > bies_probs[i, 1] * bies_probs[i+1, 1]:
                        out_bies += "eb"
                    else:
                        out_bies += "ii"
                    start_of_word = False
                    i += 2
                    continue
                elif bies_str[i+1] == 's':
                    # Decide between "es" and "ie":
                    if bies_probs[i, 2] * bies_probs[i+1, 3] > bies_probs[i, 1] * bies_probs[i+1, 2]:
                        out_bies += "es"
                    else:
                        out_bies += "ie"
                    start_of_word = True
                    i += 2
                    continue
            if bies_str[i] == 'e':
                out_bies += bies_str[i]
                start_of_word = True
                i += 1
                continue
            if bies_str[i] in ['b', 's']:
                # We took care of case "ib" and "is" before, so if we are here we have "bb" or "bs"
                if i == len(bies_str) - 1:
                    out_bies += 'e'
                    start_of_word = True
                    i += 1
                    continue
                elif bies_str[i+1] in ['i', 'b']:
                    # Decide between "ii" and "eb":
                    if bies_probs[i, 1] * bies_probs[i+1, 1] > bies_probs[i, 2] * bies_probs[i+1, 0]:
                        out_bies += "ii"
                    else:
                        out_bies += "eb"
                    start_of_word = False
                    i += 2
                    continue
                elif bies_str[i+1] in ['e', 's']:
                    # Decide between "ie" and "es":
                    if bies_probs[i, 1] * bies_probs[i+1, 2] > bies_probs[i, 2] * bies_probs[i+1, 3]:
                        out_bies += "ie"
                    else:
                        out_bies += "es"
                    start_of_word = True
                    i += 2
                    continue
    return out_bies


def get_trainable_data2(starting_text, ending_text, pseudo, graph_clust_ids):
    category = ["news", "encyclopedia", "article", "novel"]
    x_datas = []
    y_datas = []
    excess_grapheme_ids = max(graph_clust_ids.values()) + 1
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

                    # lines.append(new_line)
                    true_bies = line.get_bies("man")
                    y_data = true_bies.mat
                    line_len = len(line.char_brkpoints) - 1
                    x_data = np.zeros(shape=[line_len, 1])
                    for i in range(line_len):
                        char_start = line.char_brkpoints[i]
                        char_finish = line.char_brkpoints[i + 1]
                        curr_char = line.unsegmented[char_start: char_finish]
                        x_data[i, 0] = graph_clust_ids.get(curr_char, excess_grapheme_ids)
                    y_datas.append(y_data)
                    x_datas.append(x_data)
    return x_datas, y_datas


def test_model(self):
    """
    This function tests the model fitted in self.train(). It first divide the whole test data into big batches each
    of size self.t, and then break down each big batch intor smaller batches each of size self.n.
    """
    # Get testing data
    input_str = None
    if self.evaluation_data == "BEST":
        input_str = get_best_data_text(starting_text=40, ending_text=50, pseudo=False, exclusive=False)
    elif self.evaluation_data == "exclusive BEST":
        input_str = get_best_data_text(starting_text=40, ending_text=50, pseudo=False, exclusive=True)
    elif self.evaluation_data == "SAFT":
        file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT/test.txt')
        input_str = get_segmented_file_in_one_line(file, input_type="man_segmented", output_type="man_segmented")
    elif self.evaluation_data == "my":
        file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_test.txt')
        input_str = get_segmented_file_in_one_line(file, input_type="unsegmented", output_type="icu_segmented")
    else:
        print("Warning: no implementation for this evaluation data exists!")
    x_data, y_data = self._get_trainable_data(input_str)



    num_big_batches = len(x_data) // self.t
    if len(x_data) < self.t:
        print("Warning: length of the test data is smaller than self.t")
    accuracy = Accuracy()
    # Testing each big batch
    for k in range(num_big_batches):
        curr_x_data = x_data[k * self.t: (k + 1) * self.t]
        curr_y_data = y_data[k * self.t: (k + 1) * self.t, :]
        test_generator = KerasBatchGenerator(curr_x_data, curr_y_data, n=self.n, batch_size=self.batch_size)

        # Testing each mini batch
        all_test_input, all_actual_y = test_generator.generate_once(embedding_type=self.embedding_type)
        print(all_test_input.shape)
        print(all_test_input)
        # x = input()
        all_y_hat = self.model.predict(all_test_input, batch_size=self.batch_size)
        for i in range(self.batch_size):
            actual_y = all_actual_y[i, :, :]
            actual_y = Bies(input_bies=actual_y, input_type="mat")
            y_hat = Bies(input_bies=all_y_hat[i, :, :], input_type="mat")
            accuracy.update(true_bies=actual_y.str, est_bies=y_hat.str)

    # print("The BIES accuracy in test_model function: {}".format(accuracy.get_bies_accuracy()))
    # print("The F1 socre in test_model function: {}".format(accuracy.get_f1_score()))
    return accuracy.get_bies_accuracy()

def fast_test(self, lines):
    for line in lines:
        predict_input = []
        for ch in line:
            predict_input.append(self.codepoint_dic.get(ch, self.codepoints_num-1))
        predict_input = np.array(predict_input)
        predict_input = np.reshape(predict_input, newshape=[1, predict_input.shape[0]])
        # self.model.predict(predict_input)
        self.model(predict_input, training=False)
