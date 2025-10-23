## Model အသေးစိတ်အချက်အလက်များ

ရေးသားသူ: Sahand Farhoodi (sahandfr@gmail.com, sahand.farhoodi93@gmail.com)

ဤနေရာတွင်၊ model တစ်ခုချင်းစီရှိ hyper-parameter အမျိုးမျိုးနှင့် ဤ repository ရှိ trained model များ၏ အသေးစိတ်အချက်အလက်များကို ရှင်းပြပါမည်။ ဤအချက်အလက်များသည် ဤ repository ကို အသုံးပြု၍ ထိုင်း၊ မြန်မာ၊ သို့မဟုတ် ဘာသာစကားအသစ်တစ်ခုခုအတွက် model အသစ် train လိုသူတစ်ဦးအတွက် အလွန်အသုံးဝင်ပါလိမ့်မည်။

### Hyper-parameters
`WordSegmenter` ၏ instance အသစ်တစ်ခုကို အောက်ပါ command ဖြင့် ပြုလုပ်သောအခါ

``` python
model_name = "Thai_codepoints_exclusive_model4_heavy"
word_segmenter = WordSegmenter(input_name=model_name, input_n=300, input_t=1200000, input_clusters_num=350,
                               input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=15, input_training_data="exclusive BEST",
                               input_evaluation_data="exclusive BEST", input_language="Thai",
                               input_embedding_type="codepoints")
```

အောက်ပါ hyper-parameter များကို သတ်မှတ်ရန် လိုအပ်သည်-

*   **input_name:** ၎င်းသည် သင် train နေသော model ၏ အမည်ဖြစ်သည်။ ရိုးရှင်းသော convention တစ်ခုကို လိုက်နာပါက ဤ repository ၏ ကျန်အပိုင်းများသည် ပိုမိုချောမွေ့စွာ အလုပ်လုပ်ပါလိမ့်မည်။ ဤ convention ကို ဥပမာတစ်ခုဖြင့် ရှင်းပြပါမည်- `Thai_codepoints_exclusive_model4_heavy` ဟူသော အမည်ကို သုံးသပ်ကြည့်ပါ-

    *   ပထမအပိုင်းသည် model ၏ ဘာသာစကားဖြစ်သည်။
    *   ဒုတိယအပိုင်းသည် embedding type ကို ပြသော `codepoints` ဖြစ်ပြီး ဤနည်းအတိုင်း အသုံးပြုရန် အရေးကြီးသည်။ `input_embedding_type = codepoints` ဖြစ်ပါက `codepoints` ကို အသုံးပြုနိုင်သည်၊ `input_embedding_type = grapheme_clusters_tf` သို့မဟုတ် `input_embedding_type = grapheme_clusters_man` ဖြစ်ပါက `graphclust` ကို အသုံးပြုနိုင်သည်၊ နှင့် `generalized_vectors` ကို သင်၏ embedding အတွက် အသုံးပြုလိုပါက `genvec` ၏ မတူညီသော version များ (ဥပမာ `genvec123`, `genvec12d0`, etc) ကို အသုံးပြုနိုင်သည်။ Generalized vectors embedding ၏ မတူညီသော version များကို ဤ document တွင် နောက်ပိုင်း၌ ရှင်းပြထားသည်။
    *   အမည်၏ နောက်တစ်ပိုင်းသည် `exclusive` ဖြစ်သည်။ ၎င်းသည် သင်၏ model ကို train ရန် Thai-script-only text ကို အသုံးပြုခဲ့ခြင်း ရှိ၊ မရှိကို ပြောပြသည်။ ၎င်းသည် သင်၏ training data တွင် space, mark, သို့မဟုတ် Latin စာလုံးများ မပါဝင်ခဲ့ဟု အဓိပ္ပာယ်ရသည်။ အကယ်၍ သင် ထိုကဲ့သို့သော training data ကို အသုံးမပြုပါက သင်၏ model အမည်မှ `exclusive` ကို ဖျက်ပစ်နိုင်သည်။
    *   နောက်တစ်ပိုင်း၊ ကျွန်ုပ်တို့၏ ဥပမာတွင် `model4` သည် သင်၏ model ကို သင်ခေါ်လိုသော အမည်ဖြစ်ပြီး၊ ၎င်းသည် သင့်အပေါ် လုံးဝမူတည်သည်။ ကျွန်ုပ်၏ trained model များတွင်၊ `model5` နှင့် `model7` တို့သည် အသီးသီး အချွေတာဆုံးနှင့် အတိကျဆုံး model များကို ညွှန်ပြပြီး၊ `model4` သည် ဤနှစ်ခုကြားတွင် ရှိသော model ကို ပြသသည်။
    *   နောက်ဆုံးအပိုင်းသည် သင်၏ model ကို train ရန် data မည်မျှ အသုံးပြုခဲ့သည်ကို ပြသသည်။ ၎င်းသည် ကျွန်ုပ် နောက်ပိုင်းတွင် သတ်မှတ်မည့် `input_n` နှင့် `input_t` ၏ တန်ဖိုးများနှင့် ဆက်စပ်နေသည်။ ၎င်းသည် `heavy`, `heavier`, သို့မဟုတ် ဘာမှမပါဝင်ဘဲ ဖြစ်နိုင်သည်။

    `word_segmenter.py` ရှိ `pick_lstm_model` function ကို ကြည့်ရှုခြင်းဖြင့် ဤအမည်များကို code တွင် မည်သို့ အသုံးပြုသည်ကို သင်တွေ့နိုင်သည်။ ဥပမာအားဖြင့်၊ အောက်ပါ code သည် `heavy` နှင့် `heavier` နှင့် `input_n` နှင့် `input_t` အကြား ဆက်စပ်မှုကို ပြသသည်-

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

*   **input_n:** ၎င်းသည် model ကို train ရန် အသုံးပြုသော batch တစ်ခုစီရှိ example တစ်ခုစီရှိ grapheme cluster များ (သို့မဟုတ် သင်၏ embedding type သည် code points ဖြစ်ပါက code point များ) ၏ အရေအတွက်ဖြစ်သည်၊ သို့မဟုတ် forward နှင့် backward LSTM layer တစ်ခုစီရှိ cell အရေအတွက်ဖြစ်သည်။

*   **input_t:** ၎င်းသည် model ကို train ရန် အသုံးပြုသော grapheme cluster အားလုံး (သို့မဟုတ် embedding type သည် code points ဖြစ်ပါက code point များ) ၏ စုစုပေါင်း အရေအတွက်ဖြစ်သည်။

*   **input_clusters_num:** ၎င်းသည် embedding type သည် grapheme cluster ဖြစ်ပါက embedding layer တွင် အသုံးပြုသော grapheme cluster အရေအတွက်ဖြစ်သည်။ အကယ်၍ အခြား embedding type ကို သတ်မှတ်ထားပါက ဤ hyper-parameter သည် အခန်းကဏ္ဍမှ ပါဝင်ခြင်းမရှိပါ။

*   **input_embedding_dim:** ၎င်းသည် embedding vector တစ်ခုစီ၏ အရှည်ဖြစ်ပြီး data အရွယ်အစားနှင့် accuracy တွင် သိသာသော အခန်းကဏ္ဍမှ ပါဝင်သည်။

*   **input_hunits:** ၎င်းသည် LSTM cell တစ်ခုစီရှိ hidden unit အရေအတွက်ဖြစ်ပြီး၊ တဖန် data အရွယ်အစားနှင့် accuracy တွင် သိသာသော အခန်းကဏ္ဍမှ ပါဝင်သည်။

*   **input_dropout_rate:** ၎င်းသည် input layer ပြီးနောက်နှင့် output layer မတိုင်မီ အသုံးပြုသော dropout rate ဖြစ်သည်။

*   **input_output_dim:** ဤ hyper-parameter ကို ဤ repository တွင် အမြဲတမ်း ၄ သို့ သတ်မှတ်ထားသည်၊ အဘယ်ကြောင့်ဆိုသော် ကျွန်ုပ်တို့သည် segmented string ကို ကိုယ်စားပြုရန် BIES ကို အမြဲတမ်း အသုံးပြုသောကြောင့်ဖြစ်သည်။ Deepcut algorithm ရှိ BE ကဲ့သို့ segmented line အတွက် အခြား ကိုယ်စားပြုမှုကို အသုံးပြုသော code ကို သင် တီထွင်မှသာ ဤတန်ဖိုးကို ပြောင်းလဲရန် လိုအပ်သည်။

*   **input_epochs:** ၎င်းသည် model ကို train ရန် အသုံးပြုသော epoch အရေအတွက်ဖြစ်သည်။

*   **input_training_data/ input_evaluation_data:** ၎င်းသည် model ကို train ရန်နှင့် စမ်းသပ်ရန် အသုံးပြုသော data ဖြစ်သည်။ ထိုင်းအတွက်၊ ၎င်းသည် `"BEST"`, `"exclusive BEST"` (Thai-script-only model များအတွက်), သို့မဟုတ် `"pseudo BEST"` (ICU မှ ထုတ်လုပ်သော pseudo segmented data ကို အသုံးပြု၍ train/tested model များအတွက်) နှင့် ညီမျှနိုင်သည်။ evaluation အတွက်သာ ရရှိနိုင်သော အခြား option မှာ `"SAFT_Thai"` ဖြစ်သည်။ မြန်မာအတွက်၊ data သည် `"my"` (ICU မှ ထုတ်လုပ်သော pseudo segmented data ကို အသုံးပြု၍ train ထားသော model များအတွက်), `"exclusive my"` (`"my"` နှင့် တူညီသော်လည်း Burmese-script-only model များအတွက်), သို့မဟုတ် `"SAFT_Burmese"` (Google SAFT data ကို ရရှိနိုင်သည့်အခါ) ဖြစ်နိုင်သည်။ multilingual model များ train ရန်အတွက် `"BEST_my"` ဟူသော အခြား option လည်းရှိသည်၊ ၎င်းသည် ဤ repository တွင် လက်ရှိ အကောင်အထည်ဖော်ထားခြင်း မရှိပါ။ Training နှင့် testing အတွက် data set အမျိုးမျိုးကို အသုံးပြုခြင်းသည် ၎င်းတို့သည် compatible ဖြစ်သရွေ့ (ဘာသာစကားတစ်ခုတည်းတွင်) အဆင်ပြေသည်ကို သတိပြုပါ။ ထို့အပြင်၊ model ကို train ရန် အသုံးပြုသော data အပေါ် အခြေခံ၍ သင့်တော်သော model အမည်ကို အသုံးပြုရန် အရေးကြီးသည်၊ ဥပမာ "exclusive BEST" ကို training အတွက် အသုံးပြုပါက၊ model အမည်တွင် "Thai" နှင့် "exclusive" နှစ်ခုလုံး ပါဝင်ရမည်။

*   **input_language:** ၎င်းသည် သင်၏ model ၏ ဘာသာစကားဖြစ်သည်။

*   **input_embedding_type:** ၎င်းသည် model ကို train ရန် မည်သည့် embedding type ကို အသုံးပြုသည်ကို ဆုံးဖြတ်ပြီး၊ အောက်ပါတို့ထဲမှ တစ်ခုဖြစ်နိုင်သည်-
    *   `"grapheme_clusters_tf"`: ဤ option ကို grapheme cluster များကို embedding system အဖြစ် အသုံးပြုသည့်အခါ အသုံးပြုသင့်သည်။
    *   `"grapheme_clusters_man"`: ၎င်းသည် `"grapheme_clusters_tf"` နှင့် တူညီသော်လည်း၊ embedding layer ကို manually အကောင်အထည်ဖော်ထားသည်။ ၎င်းသည် အများအားဖြင့် embedding system အသစ်များကို စူးစမ်းလေ့လာခြင်း၏ တစ်စိတ်တစ်ပိုင်းဖြစ်ပြီး ယခုလက်တွေ့တွင် အသုံးမဝင်တော့ပါ။
    *   `"codepoints"`: ဤ option ကို embedding သည် code point များအပေါ် အခြေခံသည့်အခါ အသုံးပြုသင့်သည်။
    *   `"generalized_vectors"`: ဤ option ကို generalized vectors embedding system များထဲမှ တစ်ခုကို အသုံးပြုပါက အသုံးပြုသင့်သည်။ Generalized vectors embedding ၏ မတူညီသော version များရှိပြီး၊ version အပေါ် အခြေခံ၍ `"_12"`, `"_12"`, `"_12d0"`, `"_125"`, သို့မဟုတ် `"_1235"` တို့ထဲမှ တစ်ခုကို `"generalized_vectors"` ၏ အဆုံးသို့ ပေါင်းထည့်သင့်သည်။ ဥပမာအားဖြင့်၊ `"generalized_vectors_12"` သည် valid value ဖြစ်သည်။ ဤ version များအကြား ခြားနားချက်များအကြောင်း ပိုမိုဖတ်ရှုရန် "Embedding Discussion" ကို ဖတ်ရှုပါ။

*   **other parameters**: `WordSegmenter` ၏ `__init__` နှင့် `train_model` function များတွင် သတ်မှတ်ထားသော အခြား parameter အချို့ရှိသည်၊ ၎င်းတို့ကို code တွင် အကျဉ်းချုံး ရှင်းပြထားသည်၊ သို့မဟုတ် ကိုယ်တိုင်ရှင်းလင်းသော အမည်များရှိသည်။ အရေးကြီးဆုံးများမှာ `batch_size` နှင့် `learning_rate` ဖြစ်သည်။

`hunits` နှင့် `embedding_dim` ကို ခန့်မှန်းရန် Bayesian optimization ကို အသုံးပြုရန်အတွက်၊ `LSTMBayesianOptimization` ၏ instance တစ်ခုကို အောက်ပါ command ဖြင့် ပြုလုပ်ရန် လိုအပ်သည် (`train_thai.py` သို့မဟုတ် `train_burmese.py` မှ)-
```python
bayes_optimization = LSTMBayesianOptimization(input_language="Thai", input_n=50, input_t=10000, input_epochs=3,
                                              input_embedding_type='grapheme_clusters_tf', input_clusters_num=350,
                                              input_training_data="BEST", input_evaluation_data="BEST",
                                              input_hunits_lower=4, input_hunits_upper=64, input_embedding_dim_lower=4,
                                              input_embedding_dim_upper=64, input_c=0.05, input_iterations=10)
bayes_optimization.perform_bayesian_optimization()
```
ဤနေရာတွင် hyper-parameter အသစ်အချို့ရှိသည်ကို ရှင်းပြပါမည်-

*   **input_hunits_lower/input_hunits_upper:** ဤ parameter နှစ်ခုသည် `hunits` အတွက် domain search ကို သတ်မှတ်သည်။

*   **input_embedding_dim_lower/input_embedding_dim_upper:** ဤ parameter နှစ်ခုသည် `embedding_dim` အတွက် domain search ကို သတ်မှတ်သည်။

*   **input_c:** ဤတန်ဖိုးသည် candidate model တစ်ခုစီအတွက် cost function ကို တွက်ချက်ရာတွင် အရေးကြီးသော အခန်းကဏ္ဍမှ ပါဝင်သည်။ ၎င်းသည် ၀ နှင့် ၁ ကြား တန်ဖိုးတစ်ခု ယူသင့်ပြီး၊ model များကို သူတို့၏ အရွယ်အစားအပေါ် မည်မျှ penalize လုပ်လိုသည်ကို သတ်မှတ်သည်။ ၎င်းကို ၀ သို့ သတ်မှတ်ခြင်းဖြင့် သင်သည် အကောင်းဆုံး accuracy ရှိသော model ကို ရရှိမည်ဖြစ်ပြီး၊ (0.1, 0.2) range ရှိ တန်ဖိုးများသို့ သတ်မှတ်ခြင်းဖြင့် သင်သည် parsimonious model များကို ရရှိမည်ဖြစ်သည်။ (0, 0.1) range ရှိ တန်ဖိုးများသည် intermediate model များကို ဖြစ်ပေါ်စေသည်။

*   **input_iterations:** ဤ parameter သည် `hunits` နှင့် `embedding_dim` အတွက် အကောင်းဆုံးတန်ဖိုးကို ရှာဖွေရန် Bayesian optimization algorithm သည် model မည်မျှ fit လုပ်သင့်သည်ကို ဆုံးဖြတ်သည်။ ကျွန်ုပ်၏ model များအတွက်၊ ၁၀ အထက် တန်ဖိုးများသည် ကောင်းစွာ အလုပ်လုပ်သည်။

### Models အသေးစိတ်အချက်အလက်များ
အောက်ပါဇယားသည် model အရွယ်အစား၊ F1-score၊ နှင့် model အမျိုးမျိုးအတွက် `hunits` နှင့် `embedding_dim` အတွက် ခန့်မှန်းထားသော တန်ဖိုးများကို ပြသသည်။ ဤတန်ဖိုးများကို Bayesian optimization ကို အသုံးပြု၍ ခန့်မှန်းထားသည်။ ထိုင်းအတွက်၊ grapheme clusters embedding ဖြင့် non-exclusive model သုံးခု၊ code points embedding ဖြင့် exclusive model သုံးခု၊ နှင့် non-exclusive generalized vectors model (version 123) တစ်ခုရှိသည်။ မြန်မာအတွက်၊ grapheme clusters embedding ဖြင့် non-exclusive model သုံးခု၊ code points embedding ဖြင့် exclusive model သုံးခု၊ နှင့် non-exclusive generalized vectors model (version 1235) တစ်ခုရှိသည်။

မြန်မာ model များအတွက်၊ F1-score ကို pseudo segmented data (model type အပေါ် အခြေခံ၍ exclusive နှင့် non-exclusive) ကို အသုံးပြု၍ တွက်ချက်သည်။ ထိုင်း model များအတွက်၊ F1-score ကို BEST data set (model type အပေါ် အခြေခံ၍ exclusive နှင့် non-exclusive) ကို အသုံးပြု၍ တွက်ချက်သည်။ ဤဇယားအရ exclusive model များကို အသုံးပြုခြင်း၏ accuracy အပေါ် αρνηဘက်အကျိုးသက်ရောက်မှုသည် မြန်မာအတွက် ပိုမိုသိသာသည်ကို ကျွန်ုပ်တို့တွေ့ရသည်၊ အဘယ်ကြောင့်ဆိုသော် ၎င်းတွင် space များ ပိုမိုပါဝင်သောကြောင့် ဖြစ်နိုင်သည်။


| Model | embedding_dim | hunits | F1-score | model size |
| :---: | :----: | :---: | :---: | :---: |
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
