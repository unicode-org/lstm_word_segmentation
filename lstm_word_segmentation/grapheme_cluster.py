import numpy as np
from . import constants
from icu import Char


class GraphemeCluster:
    """
    A class that to store a grapheme cluster. It supports three versions of a grapheme cluster:
        1) graph_clust_id: an integer that is computed with respect to a dictionary of top grapheme clusters
        2) graph_clust_vec: a vector that has only one at the slot associated with graph_clust_id
        3) generalized_vec: a vector of a fixed size that has 1/m for each code point found in the grapheme cluster,
        where m is the total number of code points in the grapheme cluster
    """
    def __init__(self, grapheme_cluster, graph_clust_dic, letters_dic):
        """
        The __init__ function creates a new instance of the class.
        Args:
            grapheme_cluster: the input grapheme cluster
            graph_clust_dic: the dictionary that stores all top graph clusters in a language
            letters_dic: a dictionary that determines how different code points are mapped to different slots of
            generalized_vec.
        """
        self.graph_clust = grapheme_cluster
        self.num_clusters = len(graph_clust_dic)+1
        self.graph_clust_id = graph_clust_dic.get(self.graph_clust, self.num_clusters - 1)
        self.graph_clust_vec = np.zeros(self.num_clusters)
        self.graph_clust_vec[self.graph_clust_id] = 1
        self.num_letters = len(letters_dic)
        self.generalized_vec_length = self.num_letters + 4
        self.generalized_vec = np.zeros(self.generalized_vec_length)
        for ch in self.graph_clust:
            if constants.CHAR_TYPE_TO_BUCKET[Char.charType(ch)] == 1:
                self.generalized_vec[letters_dic.get(ch, self.num_letters)] += 1
            if constants.CHAR_TYPE_TO_BUCKET[Char.charType(ch)] in [2, 5, 6]:
                self.generalized_vec[self.num_letters + 1] += 1
            if constants.CHAR_TYPE_TO_BUCKET[Char.charType(ch)] == 3:
                self.generalized_vec[self.num_letters + 2] += 1
            if constants.CHAR_TYPE_TO_BUCKET[Char.charType(ch)] in [4, 7]:
                self.generalized_vec[self.num_letters + 3] += 1
        self.generalized_vec = self.generalized_vec/np.sum(self.generalized_vec)

    def display(self):
        """
        A function that displayes different versions of the current grapheme_cluster
        """
        print("Grapheme cluster: \n{}".format(self.graph_clust))
        print("Grapheme cluster id: \n{}".format(self.graph_clust_id))
        print("Grapheme cluster vector: \n{}".format(self.graph_clust_vec))
        print("Generalized vector: \n{}".format(self.generalized_vec))
