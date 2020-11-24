## LSTM-based model for Word Segmentation

In this project we develop a bi-directional LSTM model for word segmentation. For now, this model is implemented for Thai and Burmese.

### The model structure
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
* **Thai**: The following table summarizes the performance of our algorithm alongside with that of the state of the art algorithm [Deepcut](https://github.com/rkcosmos/deepcut) and current ICU algorithm for Thai. We have different versions of our algorithm, where some of them are designed as parsimonious as possible, and some of them are larger models but potentially with a better performance in terms of word segmentation accuracy. Based on this table, LSTM model 4 is much lighter and faster than the Deepcut, and hence is more appropriate for applications where size of the model matters, such as ????. Deepcut outperforms this model by a considerable margin on the BEST data, but for other data sets such as SAFT data, which are not the data used to train this model, this margind drops significantly. LSTM model 1 is a model of larger size, which ..... LSTM model 5 is the most parsimonious model presented in the following table, with ...

| Algorithm | BIES accuracy (BEST) | F1-score (BEST) | BIES accuracy (SAFT) | F1-score (SAFT) | Model size | Run time |
| :---:     |         :----:       |      :---:      |         :----:       |      :---:      | :---:  |   :---:  |
| LSTM (model1)  | ???? | ???? | ???? | ???? | ?? KB | ??? |
| LSTM (model4)  | 95.2 | 90.8 | 91.5 | 83.9 | 57 KB | ??? |
| LSTM (model5)  | ???? | ???? | ???? | ???? | ?? KB | ??? |
| Deepcut         | 97.8 | 95.7 | 92.6 | 836  | 2.2 MB | ??? |
| ICU             | 91.9 | 85 | 90.3 | 81.9 | 126 KB | ??? |

* **Burmese**: 




