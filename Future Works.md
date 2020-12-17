## Future Works

Author: Sahand Farhoodi (sahandfr@gmail.com, sahand.farhoodi93@gmail.com)

Here we mention some areas for improving this repository.

### Training models with variable-length input
The current LSTM models are trained using samples of fixed length (`self.n`). This means that in most cases we have to break long sentences into multiple examples. Having fixed-length samples makes training of the LSTM model easier (multiple samples can be used to update weights which makes the backpropagation algorithm much more stable), but can potentially harm the model accuracy. Another drawback of it is that at the evaluation time, eventually, we are going to use our model to segment sentences of variable lengths. Using TensorFlow prediction functions segments these sentences correctly (we checked it using our manual implementation), but it generates some warnings. Furthermore, our analysis shows that using `tf.predict` function in this case is much slower. One area for development is to write a code that uses variable-length samples for training while it still is capable of merging multiple samples to update weights of the LSTM model.

### Complete analysis of runtime
Our initial analysis implies that the current ICU algorithm is approximately 5 times faster than our models when `tf.predict` is used for evaluation. This number for Deepcut, which is considerably larger than our models, is about 39. Given that deep learning models perform a lot of matrix multiplication while ICU algorithms mostly look-up dictionaries, this difference in runtime is more-or-less expected. However, there are some ideas for making our algorithm faster that remain to be explored:
  
  * One approach is using fast versions of `sigmoid` and `tanh` functions. Based on our analysis with replacing float32 with float 16 in our moels, we don't expect a decrease in model accuracy when fast versions of `sigmoid` and `tanh` are used.
  
  * Multiplying sparse matrices is much faster. One idea is to set small values in model matrices to zero to decrease the run time. To do so, first, we need to see how small are values in our matrices, and second, make sure making matrices sparse doesn't decrease accuracy a lot.
  
  * Using fast matrix multiplication approaches that sacrifice precision a little but result in much faster algorithms is another unexplored area.

### Using other representation of segmented strings
Our LSTM models are based on representing a segmented string with BIES (B for beginning, I for inside, E for end, and S for single). Therefore, what our algorithm does during the training is maximizing BIES accuracy (the difference between estimated BIES and true BIES). Ideally, a word-segmentation training algorithm should maximize the F1-score because this is the measure that we care about at evaluation time. However, that is not doable because we cannot take derivatives of F1-score at each unit of a sentence (grapheme cluster or code point). Therefore, BIES is used as a proxy for F1-score. An interesting question is what other alternatives exist? For example, some word segmentation models use BE (B for beginning and E for not beginning). Another option is to use a representation that we propose, called BInES. The difference between BInES and BIES is that instead of representing all inside characters with I, we have I1, I2, I3, ... that shows if a character is the first I character, is it the second I character, and so on. This representation has more information in it, but more information doesn't necessarily mean more signals. It is also harder to train a model based on BInES, because the algorithm has a harder task of detecting signals from noise. This area is open to more investigations. 

### Training multilingual models
An ongoing project is training models that can segment texts in more than one language. This can be an interesting idea particularly when we have two languages, A and B, with different scripts but similar rules. In this case, at least in theory, the main difference between models trained separately for these two languages must be the embedding layer. Therefore, we can keep the number of hidden units fixed, and only increase the embedding size to train a model that can segment texts in both A and B. This will cause a dramatic decrease in the model size.

### Other embedding systems
Another area that has more room to be explored is other types of embedding. We already have investigated three embedding systems thoroughly. There are other ones, such as phonemes (see [here](https://docs.google.com/document/d/1KXnTvrgISYUplOk1NRbQbJssueeXa8k1Vu8YApMud4k/edit#heading=h.bmtbd2h7j5nt)), that can result in interesting findings.
