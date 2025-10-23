# မြန်မာဘာသာအတွက် Word Segmentation Model အသုံးပြုနည်းနှင့် တည်ဆောက်ပုံ လမ်းညွှန်

## နိဒါန်း

ဤ Repository သည် Thai (ထိုင်း) နှင့် Burmese (မြန်မာ) ဘာသာစကားများအတွက် Bi-directional LSTM (Long Short-Term Memory) model ကို အသုံးပြု၍ Word Segmentation (စကားလုံးများ ပိုင်းခြားခြင်း) ပြုလုပ်ရန် ရည်ရွယ်၍ တည်ဆောက်ထားပါသည်။ ဤလမ်းညွှန်တွင် မြန်မာဘာသာအတွက် Model ကို အသုံးပြုပုံ၊ Model အသစ်တည်ဆောက်ပုံ နှင့် Model ၏ အသေးစိတ် အချက်အလက်များကို ပြည့်စုံစွာ ရှင်းလင်းဖော်ပြသွားမည် ဖြစ်ပါသည်။

## Pre-trained Model အသုံးပြုပုံ

Repository တွင် အသင့် Train လုပ်ပြီးသား Model များကို အလွယ်တကူ အသုံးပြုနိုင်ပါသည်။ အောက်ပါအဆင့်များအတိုင်း လုပ်ဆောင်ခြင်းဖြင့် မိမိ လိုအပ်သော စာသားများကို အလွယ်တကူ ပိုင်းခြားနိုင်ပါသည်။

၁. `train_burmese.py` ဖိုင်ကို ဖွင့်ပါ။

၂. `# Choose one of the saved models to use` ဟု ရေးသားထားသော အပိုင်းကို ရှာပါ။

၃. `pick_lstm_model` function ကို အသုံးပြု၍ မိမိအသုံးပြုလိုသော Model ကို ရွေးချယ်ပါ။

```python
from lstm_word_segmentation.word_segmenter import pick_lstm_model

# အသင့်သိမ်းဆည်းထားသော Model များထဲမှ တစ်ခုကို ရွေးချယ်အသုံးပြုပါ
word_segmenter = pick_lstm_model(model_name="Burmese_graphclust_model5_heavy", embedding="grapheme_clusters_tf",
                                 train_data="my", eval_data="my")
```

**Parameter ရှင်းလင်းချက်:**
*   `model_name`: အသုံးပြုလိုသော Model ၏ အမည်။ (Repository ရှိ `Models/` folder ထဲတွင် ကြည့်ရှုနိုင်ပါသည်။)
*   `embedding`: Model ကို Train ရာတွင် အသုံးပြုခဲ့သော Embedding အမျိုးအစား။
*   `train_data`: Model ကို Train ရာတွင် အသုံးပြုခဲ့သော Data။
*   `eval_data`: Model ကို စမ်းသပ်ရာတွင် အသုံးပြုခဲ့သော Data။

**မှတ်ချက်။** Model အမည်နှင့် Embedding အမျိုးအစားသည် ကိုက်ညီမှုရှိရန် လိုအပ်ပါသည်။ Model များ၏ အသေးစိတ် အချက်အလက်များကို `Models Specifications.md` တွင် ကြည့်ရှုနိုင်ပါသည်။

၄. `segment_arbitrary_line` function ကို အသုံးပြု၍ စာကြောင်းများကို ပိုင်းခြားပါ။

```python
# ပိုင်းခြားလိုသော စာကြောင်းကို ထည့်ပါ
line = "မြန်မာဘာသာစကားသည်အလွန်လှပပါသည်။"

# Model ကို အသုံးပြု၍ စာကြောင်းကို ပိုင်းခြားပါ
segmented_line = word_segmenter.segment_arbitrary_line(line)

# ရလဒ်ကို print ထုတ်ကြည့်ပါ
print(segmented_line)
```

**နမူနာ ရလဒ်:**
```
|မြန်မာဘာသာစကား|သည်|အလွန်|လှပ|ပါ|သည်။|
```

## Model အသစ် Train ခြင်း

မိမိ၏ ကိုယ်ပိုင် Data များဖြင့် Model အသစ် Train ခြင်းကို အောက်ပါအဆင့်များအတိုင်း လုပ်ဆောင်နိုင်ပါသည်။

၁. `train_burmese.py` ဖိုင်ကို ဖွင့်ပါ။

၂. `# Train a new model -- choose name cautiously to not overwrite other models` ဟု ရေးသားထားသော အပိုင်းကို ရှာပါ။

၃. `WordSegmenter` class ကို အသုံးပြု၍ Model အသစ်တစ်ခုကို သတ်မှတ်ပါ။

```python
from lstm_word_segmentation.word_segmenter import WordSegmenter

# Model အသစ်တစ်ခုကို သတ်မှတ်ပါ
model_name = "Burmese_new_model"
word_segmenter = WordSegmenter(input_name=model_name, input_n=200, input_t=600000, input_clusters_num=350,
                               input_embedding_dim=28, input_hunits=14, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=20, input_training_data="exclusive my", input_evaluation_data="exclusive my",
                               input_language="Burmese", input_embedding_type="grapheme_clusters_tf")
```

**`WordSegmenter` Class ၏ Parameter များ:**
*   `input_name`: Train မည့် Model အသစ်၏ အမည်။
*   `input_n`: LSTM model သို့ input လုပ်မည့် sequence ၏ အလျား။
*   `input_t`: Model ကို Train ရန်နှင့် Validate လုပ်ရန် အသုံးပြုမည့် Data ၏ စုစုပေါင်း အလျား။
*   `input_clusters_num`: Grapheme cluster embedding ကို အသုံးပြုပါက ထည့်သွင်းရမည့် grapheme cluster အရေအတွက်။
*   `input_embedding_dim`: Embedding vector တစ်ခုစီ၏ အလျား။
*   `input_hunits`: LSTM cell တစ်ခုစီတွင် အသုံးပြုမည့် hidden unit အရေအတွက်။
*   `input_dropout_rate`: Dropout rate.
*   `input_output_dim`: Output layer ၏ dimension (BIES ကို အသုံးပြုသောကြောင့် ၄ ဖြစ်ပါသည်)။
*   `input_epochs`: Model ကို Train မည့် epoch အရေအတွက်။
*   `input_training_data`: Model ကို Train ရန် အသုံးပြုမည့် Data ၏ အမည်။
*   `input_evaluation_data`: Model ကို စမ်းသပ်ရန် အသုံးပြုမည့် Data ၏ အမည်။
*   `input_language`: Model ၏ ဘာသာစကား။
*   `input_embedding_type`: အသုံးပြုမည့် Embedding အမျိုးအစား။

၄. `train_model()`, `save_model()`, `test_model_line_by_line()` function များကို အသုံးပြု၍ Model ကို Train ခြင်း၊ သိမ်းဆည်းခြင်း နှင့် စမ်းသပ်ခြင်းများ ပြုလုပ်ပါ။

```python
# Model ကို Train ပါ
word_segmenter.train_model()

# Model ကို သိမ်းဆည်းပါ
word_segmenter.save_model()

# Model ကို စမ်းသပ်ပါ
word_segmenter.test_model_line_by_line(verbose=True)
```

## Dataset ပြင်ဆင်ခြင်းနှင့် BIES Tag များ

Model ၏ အရည်အသွေးသည် Training Data ၏ အရည်အသွေးပေါ်တွင် အဓိကမူတည်ပါသည်။

### Dataset အရွယ်အစားနှင့် အရည်အသွေး

*   **Dataset အရည်အသွေး (Quality):** Training data တွင် စကားလုံး ပိုင်းခြားခြင်းကို တသမတ်တည်း (consistently) ပြုလုပ်ထားရန် အလွန်အရေးကြီးပါသည်။ မမှန်ကန်သော သို့မဟုတ် တသမတ်တည်းမရှိသော segmented data များသည် model ၏ တိကျမှန်ကန်မှုကို ထိခိုက်စေနိုင်ပါသည်။
*   **Dataset အရွယ်အစား (Size):** စာကြောင်းအရေအတွက် အနည်းဆုံးဟု တိကျစွာ သတ်မှတ်ထားခြင်းမရှိပါ။ သို့သော်၊ data ပမာဏ များပြားပြီး ကွဲပြားหลากหลายလေ (diverse)၊ model ၏ အရည်အသွေး ပိုကောင်းလေ ဖြစ်ပါသည်။ `input_t` parameter သည် model ကို train ရန် အသုံးပြုမည့် စုစုပေါင်း grapheme cluster (သို့မဟုတ် codepoint) အရေအတွက်ကို ကိုယ်စားပြုပါသည်။ ယေဘုယျအားဖြင့် `input_t` တန်ဖိုး သိန်းဂဏန်း (ဥပမာ - 600,000) သည် ကောင်းမွန်သော ရလဒ်များကို ရရှိစေနိုင်ပါသည်။
*   **စာကြောင်း အရှည် (Max Character Length):** Model သည် စာကြောင်းများကို တိုက်ရိုက် train ခြင်းမဟုတ်ဘဲ၊ စာသားအားလုံးကို တစ်ဆက်တည်းဖြစ်အောင်ပေါင်းပြီး `input_n` (ဥပမာ - 200) အရွယ်အစားရှိသော အပိုင်းငယ်များ (chunks) အဖြစ် ပိုင်းခြားကာ train ခြင်းဖြစ်ပါသည်။ ထို့ကြောင့် စာကြောင်းတစ်ကြောင်း၏ အရှည်ကို ကန့်သတ်ထားခြင်း မရှိပါ။

### BIES Tag များ ရှင်းလင်းချက်

Model ၏ Output Layer သည် စာလုံး (grapheme cluster) တစ်ခုချင်းစီကို အောက်ပါ BIES tag များထဲမှ တစ်ခုခုအဖြစ် သတ်မှတ်ပေးခြင်းဖြင့် စကားလုံးများကို ပိုင်းခြားပါသည်။

*   **B (Beginning):** စကားလုံး၏ အစ စာလုံး။
*   **I (Inside):** စကားလုံး၏ အလယ် (အတွင်း) စာလုံး။
*   **E (End):** စကားလုံး၏ အဆုံး စာလုံး။
*   **S (Single):** တစ်လုံးတည်းဖြင့် ဖွဲ့စည်းထားသော စကားလုံး။

**နမူနာ:** `"မြန်မာစာ"` ဟူသော စကားလုံးကို ပိုင်းခြားခြင်း

*   **Input:** `မြန်မာစာ`
*   **Segmented Output:** `|မြန်မာ|စာ|`
*   **Grapheme Clusters:** `မြ`, `န်`, `မာ`, `စာ`
*   **BIES Tags:**
    *   `မြ` -> **B** (မြန်မာ ဆိုသော စကားလုံး၏ အစ)
    *   `န်` -> **I** (မြန်မာ ဆိုသော စကားလုံး၏ အတွင်း)
    *   `မာ` -> **E** (မြန်မာ ဆိုသော စကားလုံး၏ အဆုံး)
    *   `စာ` -> **S** (စာ ဆိုသော တစ်လုံးတည်း စကားလုံး)

ဤ BIES tag များကို အခြေခံ၍ model သည် စကားလုံးများ၏ နယ်နိမိတ်ကို သတ်မှတ်ပေးပါသည်။

## Model တည်ဆောက်ပုံ

ဤ Bi-directional LSTM model ၏ တည်ဆောက်ပုံကို အောက်ပါပုံတွင် ဖော်ပြထားပါသည်။

![Figure 1. The model structure for a bi-directional LSTM.](Figures/model_structure.png)

Model ၏ Layer တစ်ခုချင်းစီ၏ အလုပ်လုပ်ပုံမှာ အောက်ပါအတိုင်း ဖြစ်ပါသည်။

*   **Input Layer**: Input Layer တွင် စကားလုံးများ ပိုင်းခြားလိုသော စာကြောင်းကို ထည့်သွင်းပါသည်။ စာကြောင်းကို Grapheme Cluster များ သို့မဟုတ် Code Point များ၏ Sequence အဖြစ် ထည့်သွင်းနိုင်ပါသည်။
*   **Embedding Layer**: Embedding Layer တွင် Input Layer မှ ဝင်လာသော Grapheme Cluster သို့မဟုတ် Code Point တစ်ခုချင်းစီကို Model မှ နားလည်နိုင်သော ကိန်းဂဏန်း Vector များအဖြစ်သို့ ပြောင်းလဲပေးပါသည်။ Embedding အမျိုးအစား သုံးမျိုးရှိပါသည်။
    *   **grapheme clusters to vectors**: Grapheme cluster တစ်ခုချင်းစီကို Vector တစ်ခုနှင့် တွဲချိတ်ပေးပါသည်။
    *   **Generalized encoding vectors**: Code Point တစ်ခုချင်းစီကို Vector တစ်ခုနှင့် တွဲချိတ်ပြီး Grapheme Cluster တစ်ခု၏ Vector ကို ၎င်းတွင်ပါဝင်သော Code Point များ၏ Vector များ၏ ပျမ်းမျှအဖြစ် တွက်ချက်ပါသည်။
    *   **code points to vectors**: Code Point တစ်ခုချင်းစီကို Vector တစ်ခုနှင့် တိုက်ရိုက်တွဲချိတ်ပေးပါသည်။
*   **Forward/Backward LSTM Layers**: Embedding Layer မှ ရရှိလာသော Vector များကို Forward နှင့် Backward LSTM Layer များသို့ ပေးပို့ပါသည်။ LSTM Layer များသည် စာကြောင်း၏ ရှေ့နောက်ဆက်စပ်မှုကို နားလည်စေရန် ကူညီပေးပါသည်။
*   **Output Layer**: LSTM Layer များမှ ရရှိလာသော Output များကို ပေါင်းစပ်ပြီး Dense Layer သို့ ပေးပို့ပါသည်။ Dense Layer မှ Grapheme Cluster တစ်ခုချင်းစီအတွက် အထက်တွင် ရှင်းပြခဲ့သည့် BIES (Beginning, Inside, End, Single) tag များနှင့် သက်ဆိုင်သော Probability Vector တစ်ခုကို ထုတ်ပေးပါသည်။
*   **Dropout Layers**: Model ၏ Overfitting ကို လျှော့ချရန်အတွက် Dropout Layer များကို Embedding Layer ၏ နောက်နှင့် Output Layer ၏ ရှေ့တွင် ထည့်သွင်းထားပါသည်။

## Hyperparameter များ

Model ကို Train ရာတွင် အသုံးပြုသည့် အဓိက Hyperparameter များမှာ အောက်ပါအတိုင်း ဖြစ်ပါသည်။

*   `input_name`: Train မည့် Model အသစ်၏ အမည်။
*   `input_n`: LSTM model သို့ input လုပ်မည့် sequence ၏ အလျား။
*   `input_t`: Model ကို Train ရန်နှင့် Validate လုပ်ရန် အသုံးပြုမည့် Data ၏ စုစုပေါင်း အလျား။
*   `input_clusters_num`: Grapheme cluster embedding ကို အသုံးပြုပါက ထည့်သွင်းရမည့် grapheme cluster အရေအတွက်။
*   `input_embedding_dim`: Embedding vector တစ်ခုစီ၏ အလျား။ Model ၏ အရွယ်အစားနှင့် တိကျမှန်ကန်မှုအပေါ် သိသိသာသာ သက်ရောက်မှုရှိပါသည်။
*   `input_hunits`: LSTM cell တစ်ခုစီတွင် အသုံးပြုမည့် hidden unit အရေအတွက်။ ၎င်းသည်လည်း Model ၏ အရွယ်အစားနှင့် တိကျမှန်ကန်မှုအပေါ် သိသိသာသာ သက်ရောက်မှုရှိပါသည်။
*   `input_dropout_rate`: Dropout rate.
*   `input_output_dim`: Output layer ၏ dimension (BIES ကို အသုံးပြုသောကြောင့် ၄ ဖြစ်ပါသည်)။
*   `input_epochs`: Model ကို Train မည့် epoch အရေအတွက်။
*   `input_training_data` / `input_evaluation_data`: Model ကို Train ရန် နှင့် စမ်းသပ်ရန် အသုံးပြုမည့် Data ၏ အမည်။
*   `input_language`: Model ၏ ဘာသာစကား။
*   `input_embedding_type`: အသုံးပြုမည့် Embedding အမျိုးအစား။

Hyperparameter များ၏ အသေးစိတ် အချက်အလက်များကို `Models Specifications.md` တွင် အသေးစိတ် ဖတ်ရှုနိုင်ပါသည်။

## နောက်ထပ် Model Architecture တစ်ခု - CNN (Convolutional Neural Network)

ဤ Repository တွင် LSTM model အပြင်၊ ပိုမိုမြန်ဆန်သော Inference Speed ကို ရရှိနိုင်သည့် CNN model architecture ကိုလည်း ဖော်ပြထားပါသည်။

<img src="Figures/cnn.jpg"  width="30%"/>

### CNN Model ၏ အလုပ်လုပ်ပုံ

CNN model သည် စာသားများကို ပုံရိပ်များ (images) ကဲ့သို့ သဘောထားကာ အလုပ်လုပ်ပါသည်။ စာလုံး (character) တစ်ခုချင်းစီကို pixel တစ်ခုကဲ့သို့ မှတ်ယူပြီး convolution filter များဖြင့် feature များကို ထုတ်ယူပါသည်။ Dilated convolutions ကို အသုံးပြုထားသောကြောင့် ပတ်ဝန်းကျင်ရှိ စကားလုံးများ၏ context ကို ကျယ်ပြန့်စွာ ထည့်သွင်းစဉ်းစားနိုင်ပြီး တိကျမှန်ကန်မှုကို ထိန်းသိမ်းနိုင်ပါသည်။

### အားသာချက်

*   **Inference Speed:** CNN model ၏ အဓိက အားသာချက်မှာ LSTM model ထက် သိသိသာသာ ပိုမိုမြန်ဆန်သော inference speed (Segmentation ပြုလုပ်သည့် အမြန်နှုန်း) ဖြစ်ပါသည်။
*   **Time Complexity:** LSTM model ၏ O(n) time complexity ပြဿနာကို ဖြေရှင်းနိုင်ပါသည်။
*   **Accuracy:** မြန်ဆန်သော်လည်း Accuracy မှာ LSTM model နှင့် ယှဉ်နိုင်စွမ်းရှိပါသည်။

### Hyperparameters

CNN model ကို train ရန်အတွက် အောက်ပါ hyperparameter များကို အဓိကထား အသုံးပြုပါသည်။

*   `model-type`: `cnn` ဟု သတ်မှတ်ပေးရပါမည်။
*   `filters`: Conv1D layer တစ်ခုစီရှိ filter အရေအတွက်။
*   `edim`: Embedding dimension.
*   အခြား parameter များမှာ LSTM model နှင့် ဆင်တူပါသည်။

အသေးစိတ် အချက်အလက်များကို `CNN.md` ဖိုင်တွင် ဖတ်ရှုနိုင်ပါသည်။

## Debugging နှင့် Error Handling

Model ကို Train ရာတွင် အောက်ပါ Error များ နှင့် အခက်အခဲများ ကြုံတွေ့ရနိုင်ပါသည်။

| Error / အခက်အခဲ | ဖြစ်နိုင်သော အကြောင်းအရင်း | ဖြေရှင်းနည်း |
| :--- | :--- | :--- |
| `FileNotFoundError` | Training data သို့မဟုတ် testing data file များ မရှိခြင်း။ | `Data/` folder ထဲတွင် လိုအပ်သော file များ (ဥပမာ `my_train.txt`, `my_test_segmented.txt`) ရှိမရှိ စစ်ဆေးပါ။ |
| `ValueError` / `KeyError` | Model အမည် နှင့် `embedding_type` ကိုက်ညီမှု မရှိခြင်း။ | `pick_lstm_model` ကို အသုံးပြုသည့်အခါ `model_name` နှင့် `embedding` parameter မှန်ကန်မှုရှိစေရန် `Models Specifications.md` တွင် စစ်ဆေးပါ။ |
| Memory Error (`ResourceExhaustedError`) | Model အရွယ်အစား အလွန်ကြီးမားနေပြီး Memory မလုံလောက်ခြင်း။ | `input_embedding_dim` နှင့် `input_hunits` တန်ဖိုးများကို လျှော့ချပါ။ သို့မဟုတ် Memory ပိုများသော စက်တွင် Train ပါ။ |
| Accuracy နည်းပါးခြင်း | Epoch အရေအတွက် နည်းလွန်းခြင်း၊ Training data နည်းလွန်းခြင်း၊ Hyperparameter များ မသင့်တော်ခြင်း။ | `input_epochs` ကို တိုး၍ Train ကြည့်ပါ။ `input_t` တန်ဖိုးကို တိုး၍ data ပိုများများဖြင့် Train ကြည့်ပါ။ `LSTMBayesianOptimization` ကို အသုံးပြု၍ အကောင်းဆုံး Hyperparameter များကို ရှာဖွေပါ။ |
