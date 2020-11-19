import numpy as np
import constants
from icu import BreakIterator, Locale, Char, UCharCategory


class GraphemeCluster:
    def __init__(self, grapheme_cluster, graph_clust_dic, letters_dic):
        self.graph_clust = grapheme_cluster
        self.num_clusters = len(graph_clust_dic)+1
        self.graph_clust_id = graph_clust_dic.get(self.graph_clust, self.num_clusters - 1)
        self.graph_clust_vec = np.zeros(self.num_clusters)
        self.graph_clust_vec[self.graph_clust_id] = 1
        self.num_letters = len(letters_dic)
        self.generalized_vec_length = self.num_letters + 4
        self.generalized_vec = np.zeros(self.generalized_vec_length)
        for ch in self.graph_clust:
            if constants.THAI_CHAR_TYPE_TO_BUCKET[Char.charType(ch)] == 1:
                self.generalized_vec[letters_dic.get(ch, self.num_letters)] += 1
            if constants.THAI_CHAR_TYPE_TO_BUCKET[Char.charType(ch)] in [2, 5, 6]:
                self.generalized_vec[self.num_letters + 1] += 1
            if constants.THAI_CHAR_TYPE_TO_BUCKET[Char.charType(ch)] == 3:
                self.generalized_vec[self.num_letters + 2] += 1
            if constants.THAI_CHAR_TYPE_TO_BUCKET[Char.charType(ch)] in [4, 7]:
                self.generalized_vec[self.num_letters + 3] += 1
        self.generalized_vec = self.generalized_vec/np.sum(self.generalized_vec)

    def display(self):
        print("Grapheme cluster: \n{}".format(self.graph_clust))
        print("Grapheme cluster id: \n{}".format(self.graph_clust_id))
        print("Grapheme cluster vector: \n{}".format(self.graph_clust_vec))
        print("Generalized vector: \n{}".format(self.generalized_vec))

