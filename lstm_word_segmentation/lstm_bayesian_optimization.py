from .word_segmenter import WordSegmenter
from bayes_opt import BayesianOptimization


class LSTMBayesianOptimization:
    """
    A class that can be used to run Bayesian optimization algorithm to decide on the best value for parameters `hunits`
    and `embedding_dim` for bi-directional LSTM models.
    """
    def __init__(self, input_n, input_t, input_language, input_epochs, input_embedding_type, input_clusters_num,
                 input_training_data, input_evaluation_data, input_hunits_lower, input_hunits_upper,
                 input_embedding_dim_lower, input_embedding_dim_upper, input_c, input_iterations):
        """
        The __init__ function creates a new instance of the class based on the input line and its type.
        Args:
            input_language: the language of the model
            input_n: length of lines for training the LSTM model
            input_t: total length of the text to be used for training the LSTM model
            input_epochs: number of epochs to fit each individual model
            input_embedding_type: the type of embedding used to train models
            input_clusters_num: the number of top grapheme clusters (used if the embedding type is grapheme_clusters)
            input_training_data: the training data
            input_evaluation_data: the evaluation data
            input_hunits_lower and input_hunits_upper: the range of search area for hunits
            input_embedding_dim_lower and input_embedding_dim_upper: the range of search area for embedding_dim
            input_c: the constant value used in the penalty function of the lstm_score
            input_iterations: the number of iterations for Bayesian optimization algorithm
        """
        self.n = input_n
        self.t = input_t
        self.language = input_language
        self.epochs = input_epochs
        self.embedding_type = input_embedding_type
        self.clusters_num = input_clusters_num
        self.training_data = input_training_data
        self.evaluation_data = input_evaluation_data
        self.hunits_lower = input_hunits_lower
        self.hunits_upper = input_hunits_upper
        self.embedding_dim_lower = input_embedding_dim_lower
        self.embedding_dim_upper = input_embedding_dim_upper
        self.c = input_c
        self.iterations = input_iterations

        # Setting self.lambda to the number of the parameters of the largest possible model
        word_segmenter = WordSegmenter(input_name="temp", input_n=50, input_t=10000,
                                       input_clusters_num=self.clusters_num,
                                       input_embedding_dim=self.embedding_dim_upper, input_hunits=self.hunits_upper,
                                       input_dropout_rate=0.2, input_output_dim=4, input_epochs=1,
                                       input_training_data=self.training_data,
                                       input_evaluation_data=self.evaluation_data, input_language=self.language,
                                       input_embedding_type=self.embedding_type)
        self.lam = word_segmenter.model.count_paramsx()

    def lstm_score(self, hunits, embedding_dim):
        """
        Given the number of hidden units and embedding dimension, this function computes a score for a bi-directional
        LSTM which is the accuracy of the model minus a weighted penalty function linear in number of parameters
        Args:
            hunits: number of LSTM cells in bi-directional LSTM model
            embedding_dim: length of output of the embedding layer
        """
        hunits = int(round(hunits))
        embedding_dim = int(round(embedding_dim))
        word_segmenter = WordSegmenter(input_name="temp", input_n=self.n, input_t=self.t,
                                       input_clusters_num=self.clusters_num, input_embedding_dim=embedding_dim,
                                       input_hunits=hunits, input_dropout_rate=0.2, input_output_dim=4,
                                       input_epochs=self.epochs, input_training_data=self.training_data,
                                       input_evaluation_data=self.training_data, input_language=self.language,
                                       input_embedding_type=self.embedding_type)
        word_segmenter.train_model()
        return word_segmenter.test_model_line_by_line(verbose=False).get_f1_score() - self.c * self.lam * \
            word_segmenter.model.count_params()

    def perform_bayesian_optimization(self):
        """
        This function implements uses the socre function given in `lstm_score` to search for the best value for
        parameters hunits and embedding_dim.
        """
        bounds = {'hunits': (self.hunits_lower, self.hunits_upper),
                  'embedding_dim': (self.embedding_dim_lower, self.embedding_dim_upper)}
        optimizer = BayesianOptimization(f=self.lstm_score, pbounds=bounds, random_state=1)
        optimizer.maximize(init_points=2, n_iter=self.iterations)
        print(optimizer.max)
        print(optimizer.res)
