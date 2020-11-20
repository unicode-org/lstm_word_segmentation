from lstm_word_segmentation.helpers import diff_strings


class Accuracy:
    """
    A class that helps computing accuracy for word segmentation. It let you compute both BIES accuracy and F1 score.
    """
    def __init__(self):
        """
        The __init__ function creates a new instance of the class.
        """
        self.bies_length = 0
        self.bies_mismatch = 0
        self.segmented_words = 0
        self.true_words = 0
        self.correctly_segmented_words = 0

    def update(self, true_bies, est_bies):
        """
        This function updates the accuracy with respect to a new segmented line and its true segmentation.
        Args:
            true_bies: a string that indicates the true BIES sequence
            est_bies: a string that indicates the estimated BIES sequence
        """
        if len(true_bies) != len(est_bies):
            print("Warning! length of true_bies and est_bies are different!")
        self.bies_length += len(true_bies)
        self.bies_mismatch += diff_strings(true_bies, est_bies)

        true_word_brkpoints = []
        for i in range(len(true_bies)):
            if true_bies[i] in ['b', 's']:
                true_word_brkpoints.append(i)
        true_word_brkpoints.append(len(true_bies))
        est_word_brkpoints = []
        for i in range(len(est_bies)):
            if est_bies[i] in ['b', 's']:
                est_word_brkpoints.append(i)
        est_word_brkpoints.append(len(est_bies))
        true_words_tuples = [(true_word_brkpoints[i], true_word_brkpoints[i + 1]) for i in
                             range(len(true_word_brkpoints) - 1)]
        est_words_tuples = [(est_word_brkpoints[i], est_word_brkpoints[i + 1]) for i in
                            range(len(est_word_brkpoints) - 1)]
        self.true_words += len(true_words_tuples)
        self.segmented_words += len(est_words_tuples)
        self.correctly_segmented_words += len(set(est_words_tuples).intersection(true_words_tuples))

    def merge_accuracy(self, other):
        """
        This function lets us to use another Accuracy instance to update the accuracy of the current instance
        Args:
            other: the other Accuracy instance
        """
        self.segmented_words += other.segmented_words
        self.true_words += other.true_words
        self.correctly_segmented_words += other.correctly_segmented_words
        self.bies_length += other.bies_length
        self.bies_mismatch += other.bies_mismatch

    def get_f1_score(self):
        """
        This function returns the F1 score
        """
        # Computing precision
        precision = 0
        if self.segmented_words != 0:
            precision = self.correctly_segmented_words / self.segmented_words

        # Computing recall
        recall = 0
        if self.true_words != 0:
            recall = self.correctly_segmented_words / self.true_words

        # Computing F1 score
        f1 = 0
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        return f1

    def get_bies_accuracy(self):
        """
        This function returns the BIES accuracy
        """
        bies_acc = 0
        if self.bies_length != 0:
            return 1 - self.bies_mismatch/self.bies_length
        return bies_acc
