## Model Specifications

Author: Sahand Farhoodi (sahandfr@gmail.com, sahand.farhoodi93@gmail.com)

Here, I explain what are different hyper-parameters in each model and what are the specification of trained models in this repository. This information will be very useful for someone who wants to use this repository to train a new model, either in Thai, Burmese, or a new language.

### Hyper-parameters
When you make a new instance of `WordSegmenter` using command

``` python
model_name = "Thai_codepoints_exclusive_model4_heavy"
word_segmenter = WordSegmenter(input_name=model_name, input_n=300, input_t=1200000, input_clusters_num=350,
                               input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=15, input_training_data="exclusive BEST",
                               input_evaluation_data="exclusive BEST", input_language="Thai",
                               input_embedding_type="codepoints")
```

you need to specify the following hyper-parameters:

* **input_name:** This is name of the model that you are training. The remaining of this repository works much smoother if you follow a simple convention. I explain this convention by an example: consider name `Thai_codepoints_exclusive_model4_heavy`. 

    * The first part is the language of the model. 
    * The second part is `codepoints` which shows the embedding type and is important to be used in this way. You can use `codepoints` if the `input_embedding_type = codepoints`, `graphclust` if `input_embedding_type = grapheme_clusters_tf` or `input_embedding_type = grapheme_clusters_man`, and different versions of `genvec` (such as `genvec123`, `genvec12d0`, etc) if you wish to use `generalized_vectors` for your embedding. The different versions of generalized vectros embedding are explained later in this document.
    * The next part of the name is `exclusive`. This tells you if you have used Thai-script-only text to train your model or not. Note that it means that you had no spaces, marks, or Latin letters in your training data. If you don't use such training data then it is okay to erase `exclusive` from your model name. 
    * The next part, `model4` in our example, is what you want to call your model, and is totally up to you. In my trained modes, `model5` and `model7` indicate the most parsimonious and accurate models respectively, and `model4` shows a model that is somewhere between these two. 
    * The last part shows how much data you used to train your model. This ties to the values of `input_n` and `input_t` that I define later. It can be `heavy`, `heavier`, or nothing. 
  
  You can take a look at `pick_lstm_model` function in `word_segmenter.py` to see how these names are used in the code. For example, the following code shows the relation between `heavy` and `heavier` and `input_n` and `input_t`.
  
  ```python
  input_n = None
    input_t = None
    if "genvec" in model_name or "graphclust" in model_name:
        input_n = 50
        input_t = 100000
        if "heavy" in model_name:
            input_n = 200
            input_t = 600000
        elif "heavier" in model_name:
            input_n = 200
            input_t = 2000000
    elif "codepoints" in model_name:
        input_n = 100
        input_t = 200000
        if "heavy" in model_name:
            input_n = 300
            input_t = 1200000
  ```

* **input_n:** This is the number of grapheme clusters (or code points if your embedding type is code points) in each example in each batch used to train the model, or simply the number of cells in each one of forward and backward LSTM layers. 

* **input_t:** This is the total number of all grapheme clusters (or code points if embedding type is code points) used to train the model.

* **input_clusters_num:** This is the number of grapheme clusters used in the embedding layer if the embedding type is grapheme clusters. If another embedding type is specified then this hyper-parameter plays no role.

* **input_embedding_dim:** This is the length of each embedding vector and plays a significant role in data size and accuracy.

* **input_hunits:** This is the number of hidden units in each LSTM cell, and again, plays a significant role in data size and accuracy.

* **input_dropout_rate:** This is the dropout rate used after the input layer and also before the output layer.

* **input_output_dim:** This hyper-parameter is always set to 4 in this repository because we always use BIES to represent a segmented string. This value needs to be changed only if you develop some code that uses another representation for a segmented line, such as BE in the Deepcut algorithm.

* **input_epochs:** This is the number of epochs used to train the model.

* **input_training_data/ input_evaluation_data:** This is the data used to train and test the model. For Thai, it can be equal to `"BEST"`, `"exclusive BEST"` (for Thai-script-only models), or `"pseudo BEST"` (for models trained/tested using pseudo segmented data generated by ICU). Another option that is only available for evaluation is `"SAFT_Thai"`. For Burmese, the data can be `"my"` (for models trained using the pseudo segmented data generated by ICU), `"exclusive my"` (same as `"my"` but for Burmese-script-only models), or `"SAFT_Burmese"` (when Google SAFT data is accessible). There is also another option, `"BEST_my"`, which is for training multilingual models and is not implemented currently in this repository. Note that it is okay to use different data sets for training and testing as long as they are compatible (both in the same language). Also, it is important to use an appropriate model name based on the data used to train the model, e.g. if "exclusive BEST" is used for training, the model name must have both "Thai" and "exclusive" in it.

* **input_language:** This is the language of your model.

* **input_embedding_type:** This determines what type of embedding is used to train the model, and can be one of the followings:
  * `"grapheme_clusters_tf"`: This option should be used when one uses grapheme clusters as the embedding system.
  * `"grapheme_clusters_man"`: This is the same as `"grapheme_clusters_tf"`, but the embedding layer is implemented manually. This was mostly part of exploring new embedding systems and have no practical use now.
  * `"codepoints"`: this option should be used when the embedding is based on code points.
  * `"generalized_vectors"`: This should be used if one of the generalized vectors embedding systems is used. There are different versions of generalized vectors embedding, and based on the version one of `"_12"`, `"_12"`, `"_12d0"`, `"_125"`, or `"_1235"` should be added to the end of `"generalized_vectors"`. For example, a valid value would be `"generalized_vectors_12"`. Please refer to "Embedding Discussion" to read more about the difference between these versions.

* **other parameters**: There are some other parameters defined in the `__init__` and `train_model` functions of `WordSegmenter` that are explained briefly in the code, or have self-explanatory names. The most important ones are `batch_size` and `learning_rate`.

For using Bayesian optimization to estimate `hunits` and `embedding_dim`, you need to make an instance of `LSTMBayesianOptimization` using the following command (from `train_thai.py` or `train_burmese.py`):
```python
bayes_optimization = LSTMBayesianOptimization(input_language="Thai", input_n=50, input_t=10000, input_epochs=3,
                                              input_embedding_type='grapheme_clusters_tf', input_clusters_num=350,
                                              input_training_data="BEST", input_evaluation_data="BEST",
                                              input_hunits_lower=4, input_hunits_upper=64, input_embedding_dim_lower=4,
                                              input_embedding_dim_upper=64, input_c=0.05, input_iterations=10)
bayes_optimization.perform_bayesian_optimization()
```
There are a few new hyper-parameters here that I explain:

* **input_hunits_lower/input_hunits_upper:** These two parameters specify the domain search for `hunits`.

* **input_embedding_dim_lower/input_embedding_dim_upper:** These two parameters specify the domain search for `embedding_dim`.

* **input_c:** This value plays an important role in computing the cost function for each candidate model. It should take a value between 0 and 1, and identifies how much we want to penalize models based on their size. By setting it to 0 you will get the model with the best accuracy, and by setting it to values in the range (0.1, 0.2) you will get parsimonious models. Values in the range (0, 0.1) result in intermediate models.

* **input_iterations:** This parameter determines how many different models the Bayesian optimization algorithm should fit to find the best value for `hunits` and `embedding_dim`. For my models, values above 10 work fine.

### Models Details
The following table shows model size, F1-score, and estimated values for `hunits` and `embedding_dim` for different models. These values are estimated using Bayesian optimization. For Thai, we have three non-exclusive models with grapheme clusters embedding, three exclusive models with code points embedding, and one non-exclusive generalized vectors model (version 123). For Burmese, we have three non-exclusive models with grapheme clusters embedding, three exclusive models with code points embedding, and one non-exclusive generalized vectors model (version 1235).

For Burmese models, the F1-score is computed using the pseudo segmented data (exclusive and non-exclusive based on the model type). For Thai models, the F1-score is computed using BEST data set (exclusive and non-exclusive based on the model type). Based on this table we see that the negative effect of using exclusive models on accuracy is much more noticeable for Burmese, probably because it has more spaces in it.


| Model | embedding_dim | hunits | F1-score | model size |
| :---: | :----:  | :---:  | :---: | :---: |
| Thai_graphclust_model4_heavy | 16 | 23 | 89.9 | 27 KB |
| Thai_graphclust_model5_heavy | 15 | 12 | 86.6 | 10 KB |
| Thai_graphclust_model7_heavy | 29 | 47 | 91.9 | 86 KB |
| Thai_codepoints_exclusive_model4_heavy | 40 | 27 | 90.1 | 36 KB |
| Thai_codepoints_exclusive_model5_heavy | 20 | 15 | 86.7 | 12 KB |
| Thai_codepoints_exclusive_model7_heavy | 34 | 58 | 91.3 | 93 KB |
| Thai_genvec123_model5_heavy | 22 | 20 | 85.4 | 19 KB |
| Burmese_graphclust_model4_heavy | 28 | 14 | 92.9 | 30 KB |
| Burmese_graphclust_model5_heavy | 12 | 12 | 91.1 | 15 KB |
| Burmese_graphclust_model7_heavy | 54 | 44 | 94.9 | 125 KB |
| Burmese_codepoints_exclusive_model4_heavy | 40 | 27 | 85.7 | 45 KB |
| Burmese_codepoints_exclusive_model5_heavy | 20 | 15 | 82.3 | 17 KB |
| Burmese_codepoints_exclusive_model7_heavy | 29 | 47 | 87.8 | 70 KB |
| Burmese_genvec1235_model4_heavy | 33 | 20 | 90.3 | 29 KB |



