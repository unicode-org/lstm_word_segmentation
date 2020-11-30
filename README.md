## LSTM-based model for Word Segmentation

In this project we develop a bi-directional LSTM model for word segmentation. For now, this model is implemented for Thai and Burmese.

### Quick start
* **Use a pre-trained model:** To segment an arbitrary line go to file `train_language.py` where `language` is the language you want to use. For example, if your line is in *Thai*, you should use file `train_thai.py`. Over there, find comment `# Choose one of the saved models to use`. Everything before this line is for training a new model and can be commented for now. After this comment, you can use variable `model_name`to specify which already fitted model you want to use. List of all fitted models can be found under folder *Models* in this repository. After choosing your model, you can specify the type of embedding to be used when you create an instance of `WordSegmenter` that stores the fitted model:
  ```python
  word_segmenter = WordSegmenter(input_name=model_name, input_n=input_n, input_t=input_t,
                               input_clusters_num=input_clusters_num, input_embedding_dim=input_embedding_dim,
                               input_hunits=input_hunits, input_dropout_rate=0.2, input_output_dim=4, input_epochs=15,
                               input_training_data="BEST", input_evaluating_data="BEST", input_language="Thai",
                               input_embedding_type="grapheme_clusters_tf")
  word_segmenter.set_model(model)
  # word_segmenter.test_model()
  # word_segmenter.test_model_line_by_line()
  ```
The commented lines test the picked model using large data sets, and are not needed for segmenting an arbitrary line. Next, you can use the following lines of the `train_language.py` file to specify your input and segmentt it:
  ```python
  line = "ทำสิ่งต่างๆ ได้มากขึ้นขณะที่อุปกรณ์ล็อกและชาร์จอยู่ด้วยโหมดแอมเบียนท์"
  word_segmenter.segment_arbitrary_line(line)
  ```
By running these two lines of code, you get a segmentation using ICU, the LSTM algorithm you picked, and Deepcut algorithm. 

* **Train a new model:** In order to train a new model you need to use the file `train_language.py` where 'language' is the language you want to use. For example, if your line is in *Thai*, you should use file `train_thai.py`. Over there, you need to use the code below comment `# Train a new model -- choose name cautiously to not overwrite other models` and above comment `# Choose one of the saved models to use`. You need to specify a name for your new model using the variable `model_name`, and then you can specify hyperparameters of your model, embedding type, and data sets to be used for training, validation, and test by making an instance of `WordSegmenter`:
  ```python
  # Train a new model -- choose name cautiously to not overwrite other models
  model_name = "Thai_temp_genvec"
word_segmenter = WordSegmenter(input_name=model_name, input_n=50, input_t=100000, input_clusters_num=350,
                               input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=20, input_training_data="BEST", input_evaluating_data="BEST",
                               input_language="Thai", input_embedding_type="grapheme_clusters_tf")

  ```
Next, you will use `word_segmenter.train_model()` to train your model, `word_segmenter.test_model_line_by_line()` to test your model, and `word_segmenter.save_model()` to save the trained model:
  ```python
  word_segmenter.train_model()
  # word_segmenter.test_model()
  word_segmenter.test_model_line_by_line()
  word_segmenter.save_model()
  ```  
As you can see line `word_segmenter.test_model()` is commented above, because this is a funciton that tests the model in a slightly different way and was used mostly for debugging. We don't recommend using it for testing performance of your model.


### Model structure
Figure 1 illustrates our bi-directional model structure. Below we explain what are different layers:

* **Input Layer**: We set [extended grapheme clusters](https://unicode.org/reports/tr29/) as the smallest units in a word. Therefore, the input layer is a sequence of extended grapheme clusters, where each one of these grapheme clusters can have few code points in it. We expect a word segmentation algorithm to not put a word boundary in the middle of a grapheme cluster, and hence by using grapheme clusters as the smallest units of a word, this can be guaranteed. We use [ICU](https://unicode-org.github.io/icu-docs/apidoc/released/icu4c/classicu_1_1BreakIterator.html) to extract grapheme clusters of a given word.

* **Embedding Layer**: In the embedding layer, we have to represent each grapheme cluster with a numerical vector so it can be trained by the model. The choice of embedding can significantly affect the model size and performance. In this repository, two types of embedding are implemented:
  * **Map grapheme clusters to vectors**: In this approach, each grapheme cluster is mapped to a single vector. These vectors are trained with the model and need to be stored to be used later for evaluation. Given that the set of possible grapheme clusters is theoretically infinity, we cannot store one vector for each possible grapheme clusters. Hence, we use larg corpora to extract all grapheme clusters that actually happen in texts. Then we sort these grapheme clusters based on their frequency in the corpora, and store one vector for those grapheme clusters that cover 99% of the text, and one vector for any other grapheme cluster. Using this approach, we need to store about 350 grapheme cluster vectors for Thai and Burmese. This option can be used by setting `input_embedding_type` to `grapheme_clusters_tf` or `grapheme_clusters_man`.
  * **Generalized encoding vecctors**: In this approach, each code point is mapped to a vector that is learned during training, and then the vector computed for a grapheme cluster is the average of vectors corresponding to code points in that grapheme cluster. The number of code points in a language is fixed and considerably less than the number of grapheme clusters, and hence the embedding matrix will have a smaller size using this approach. There are variations of this appraoch, where instead of having one vector for each code point, we can have one vector for a group of code points that we beilieve behave similarly, such as digits. This option can be used by setting `input_embedding_type` to `generalized_vectors`.

* **Forward/Backward LSTM Layers**: The output of the embedding layer is fed into the forward and backward LSTM layers. *hunits* shows the number of hidden units in each cell of LSTM layers.

* **Output Layer**: In the output layer, the output of forward and backward lstm layers are concatenated and fed into a dense layer with *softmax* activation layer to make a vector of length four for each grapheme cluster. The values in each vector add up to 1 and are probability of *BIES*, where:
  * *B* stands for begining of the word.
  * *I* stands for inside of the word.
  * *E* stands for end of the word.
  * *S* stands for a single grapheme cluster that forms a word by itself.

* **Droput Layers**: We have two dropout layers in our model: one right after the embedding layer and one right before the output layer.


![Figure 1. The model structure for a bi-directional LSTM.](Figures/model_structure.png)

### Estimating hyperparameters of the model
There are many hyperparameters in the model that need to be estimated before using it. Among different hyper parameters, two affect the model size and performance more significantly: *hunits* and *embedding size*. We first use stepwise grid-search to decide on all hyper parameters except these two such as *learning rate*, *batch size*, *dropout rate*, etc. After that, we use [Bayesian optimization](https://github.com/fmfn/BayesianOptimization) to decide on *hunits* and *embedding size*.

### Data sets
For some languages, there are manually annotated data sets that can be used to train learning-based models, such as our model. However, for some other langauges such dataset doesn't exist. We develop a framework that let us train our model in both scenarios. In this framework (shown in Figure 2), if such dataset exist then we use it directly to train our model. However, if such data set doesn't exist, we use one of the existing algorithms, such as current ICU algorithm, to make a pseudo segmented data, and then use that pseudo segmented data to train our model. We use ICU specifically because it already supports word segmentation for almost all languages, it is light, fast, and has an acceptable performance. However, for some specific languages that better word segmentation algorithm exist, ICU can be replaced. Our analysis shows that in the absence of a segmented data set, our algorithm is capable of learning what ICU does, and in fact, it usually can outperform ICU itself. Below we explain the data sets used to train and test models:

* **Thai**: We use [NECTEC BEST data set](https://thailang.nectec.or.th/downloadcenter/index4c74.html?option=com_docman&task=cat_view&gid=42&Itemid=61) to train our model. The text files in this dataset use UTF-8 encoding and are manually segmented. There are four different genre of texts in this data set: novel, news, encyclopedia, and article. For testing the model, we use both NECTEC BEST data set and Google SAFT data set.
* **Burmese**: For Burmese, we use the [Google corpus crawler](https://github.com/google/corpuscrawler) to collect unsegmented texts, and then use ICU to generate a pseudo segmented data set to be used for training. For testing, we use both such pseudo segmented texts and SAFT data.

![Figure 2. The framework for training and testing the model.](Figures/framework.png)

### Performance summary
* **Thai**: The following table summarizes the performance of our algorithm alongside with that of the state of the art algorithm [Deepcut](https://github.com/rkcosmos/deepcut) and current ICU algorithm for Thai. We have different versions of our algorithm, where LSTM model 7 and LSTM model 5 are respectively the most accurate and the most parsimonious LSTM-based models. LSTM model 4 lies somewhere between these two models, and provides a high accuracy while still has a small data size. Based on the following table, Deepcut is by far the largest and slowest model which makes applications of it limited. The LSTM models (particularly models 14 and 5) are substantially smaller, and thereofe are more appropriate for applications where size of the model matters such as mobile applications and IoT devices. Deepcut outperforms all other methods by a considerable margin on the BEST data. However, for other data sets such as SAFT data, which are not used to train this model, this margin drops significantly.

| Algorithm | BIES accuracy (BEST) | F1-score (BEST) | BIES accuracy (SAFT) | F1-score (SAFT) | Model size | Run time |
| :---:     |         :----:       |      :---:      |         :----:       |      :---:      | :---:  |   :---:  |
| LSTM (model4)  | 95.1 | 90.8 | 91.5 | 83.9 | 57 KB | ??? |
| LSTM (model5) -- change for heavily trained | 90.5 | 82.8 | 86.9 | 76.2 | 25 KB | ??? |
| LSTM (model7)  |  96  | 92.4 | 92 | 84.9 | 180 KB | ??? |
| Deepcut         | 97.8 | 95.7 | 92.6 | 86  | 2.2 MB | ??? |
| ICU             | 91.9 | 85 | 90.3 | 81.9 | 126 KB | ??? |

* **Burmese**: 
The following table summarizes the performance of our algorithm and current ICU algorithm for Burmese. Just like Thai, we have different versions of our LSTM-based algorithm, where LSTM model 7 and LSTM model 5 are respectively the most accurate and the most parsimonious LSTM-based models. LSTM model 4 lies somewhere between these two models, and provides a high accuracy while still has a small data size. Based on this table, 

| Algorithm | BIES accuracy (ICU segmented) | F1-score (ICU segmented) | BIES accuracy (SAFT) | F1-score (SAFT) | Model size | Run time |
| :---:     |         :----:                |      :---:               |     :---:  |   :---: | :---: |   :---:  |
| LSTM (model4) | 94.7 | 92.9 | 91.7 | 90.5 | 61 KB  | ??? |
| LSTM (model5) -- change for heavily trained | 92.2 | 89.5 | 91.1 | 89.8 | 28 KB | ??? |
| LSTM (model7) | 96.2 | 94.9 | 92.3 | 91.1 | 254 KB | ??? |
| ICU            | 1 | 1 | 93.1 | 92.4 | 474 KB | ??? |



