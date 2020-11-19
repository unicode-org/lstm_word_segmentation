import constants


class LstmBayesianOptimization:
    def __init__(self, input_language, input_epochs, input_embedding_type, input_clusters_num, input_hunits_lower,
                 input_hunits_upper, input_embedding_dim_lower, input_embedding_dim_upper):
        self.language = input_language
        self.epochs = input_epochs
        self.embedding_type = input_embedding_type
        self.clusters_num = input_clusters_num
        self.hunits_lower = input_hunits_lower
        self.hunits_upper = input_hunits_upper
        self.embedding_dim_lower = input_embedding_dim_lower
        self.embedding_dim_upper = input_embedding_dim_upper


        if self.language == "Thai":
            self.training_data = "BEST"
            self.evaluating_data = "BEST"


    def lstm_score(self, hunits, embedding_dim):
        """
        Given the number of LSTM cells and embedding dimension, this function computes a score for a bi-directional LSTM
        model which is basically the accuracy of the model minus a weighted penalty function linear in number of parameters
        Args:
            hunits: number of LSTM cells in bi-directional LSTM model
            embedding_dim: length of output of the embedding layer
        """
        hunits = int(round(hunits))
        embedding_dim = int(round(embedding_dim))
        word_segmenter = WordSegmenter(input_name="temp", input_n=50, input_t=100000, input_clusters_num=self.clusters_num,
                                       input_embedding_dim=embedding_dim, input_hunits=hunits, input_dropout_rate=0.2,
                                       input_output_dim=4, input_epochs=self.epochs, input_training_data=self.training_data,
                                       input_evaluating_data=self.training_data, input_language=self.language,
                                       input_embedding_type=self.embedding_type)
        word_segmenter.train_model()
        fitted_model = word_segmenter.get_model()
        lam = 1/88964  # This is number of parameters in the largest model
        C = 0.05
        return word_segmenter.test_model() - C * lam * fitted_model.count_params()


    def perform_bayesian_optimization(self):
        """
        This function implements Bayesian optimization to search in a range of possible values for number of LSTM cells and
        embedding dimension to find the most accurate and parsimonious model. It uses the function LSTM_score to compute
        score of each model.
        Args:
            hunits_lower and hunits_upper: lower and upper bound of search region for number of LSTM cells
            embedding_dim_lower and embedding_dim_upper: lower and upper bound of search region for embedding dimension
        """
        bounds = {'hunits': (self.hunits_lower, self.hunits_upper), 'embedding_dim': (self.embedding_dim_lower,
                                                                                      self.embedding_dim_upper)}
        optimizer = BayesianOptimization(
            f=self.lstm_score,
            pbounds=bounds,
            random_state=1,
        )
        optimizer.maximize(init_points=2, n_iter=10)
        print(optimizer.max)
        print(optimizer.res)

