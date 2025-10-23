## Word Segmentation အတွက် LSTM-based Model
ရေးသားသူ: Sahand Farhoodi (sahandfr@gmail.com, sahand.farhoodi93@gmail.com)

ဤ project တွင်၊ ကျွန်ုပ်တို့သည် word segmentation အတွက် bi-directional LSTM model တစ်ခုကို တီထွင်ထားပါသည်။ ယခုအချိန်တွင်၊ ဤ model များကို Thai (ထိုင်း) နှင့် Burmese (မြန်မာ) ဘာသာစကားများအတွက် train လုပ်ထားပါသည်။

### အမြန်စတင်အသုံးပြုနည်း
*   **Pre-trained model ကို အသုံးပြုခြင်း:** စာကြောင်းတစ်ကြောင်းကို segment လုပ်ရန်၊ သင်အသုံးပြုလိုသော language နှင့် သက်ဆိုင်သော `train_language.py` ဖိုင်သို့ သွားပါ။ ဥပမာအားဖြင့်၊ အကယ်၍ စာကြောင်းသည် ထိုင်းဘာသာစကားဖြစ်ပါက `train_thai.py` ဖိုင်ကို အသုံးပြုသင့်ပါသည်။ ထိုဖိုင်ထဲတွင် `# Choose one of the saved models to use` ဟူသော comment ကို ရှာပါ။ ဤ comment မတိုင်မီက code များအားလုံးသည် model အသစ် train ရန်ဖြစ်ပြီး လျစ်လျူရှုနိုင်ပါသည်။ ဤ comment ပြီးနောက်၊ `pick_lstm_model` function ကို အသုံးပြု၍ segmentation အတွက် သင်အသုံးပြုလိုသော model ကို ရွေးချယ်နိုင်သည်-

    ```python
    word_segmenter = pick_lstm_model(model_name="Thai_codepoints_exclusive_model4_heavy", embedding="codepoints",
                                   train_data="exclusive BEST", eval_data="exclusive BEST")
    ```

    `embedding`, `train_data`, နှင့် `eval_data` ဟူသော hyper-parameter သုံးခုကို သတ်မှတ်ရန် လိုအပ်ပါသည်။ ဤ hyper-parameter များ၏ အသေးစိတ်ရှင်းလင်းချက်နှင့် ဤ repository တွင် အသုံးပြုရန် အသင့်ဖြစ်နေသော trained model များစာရင်းနှင့် သူတို့၏ အသေးစိတ်အချက်အလက်များအတွက် [Models Specifications](Models_Specifications_Burmese.md) ကို ဖတ်ရှုပါ။ အကယ်၍ အချိန်မရှိပါက၊ trained model များထဲမှ တစ်ခုကို ရွေးချယ်ပြီး သင်ရွေးချယ်လိုက်သော embedding အမည်သည် model အမည်တွင် ပါဝင်နေစေရန် သေချာပါစေ (`train_data` နှင့် `eval-data` သည် input စာကြောင်းများ၏ segmentation အပေါ် သက်ရောက်မှုမရှိပါ)။ ထို့နောက်၊ သင်၏ input ကို သတ်မှတ်ပြီး segment လုပ်ရန် အောက်ပါ command များကို အသုံးပြုနိုင်သည်-

    ```python
    line = "ทำสิ่งต่างๆ ได้มากขึ้นขณะที่อุปกรณ์ล็อกและชาร์จอยู่ด้วยโหมดแอมเบียนท์"
    word_segmenter.segment_arbitrary_line(line)
    ```

*   **Model အသစ် Train ခြင်း:** ထိုင်း သို့မဟုတ် မြန်မာဘာသာဖြင့် model အသစ် train ရန်၊ သင်အလုပ်လုပ်လိုသော language နှင့် သက်ဆိုင်သော `train_language.py` ဖိုင်ကို အသုံးပြုရန် လိုအပ်ပါသည်။ ထိုဖိုင်တွင်၊ `# Train a new model -- choose name cautiously to not overwrite other models` နှင့် `# Choose one of the saved models to use` comment များကြားရှိ code ကို အသုံးပြုရန် လိုအပ်ပါသည်။ အောက်ပါ code သည် model အသစ်တစ်ခုကို သတ်မှတ်နိုင်စေသည်-

    ```python
    model_name = "Thai_new_model"
    word_segmenter = WordSegmenter(input_name=model_name, input_n=50, input_t=10000, input_clusters_num=350,
                                 input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
                                 input_epochs=1, input_training_data="exclusive BEST",
                                 input_evaluation_data="exclusive BEST", input_language="Thai",
                                 input_embedding_type="codepoints")
    ```

    အသေးစိတ်ရှင်းပြထားသော hyperparameter အချို့ကို [Models Specifications](Models_Specifications_Burmese.md) တွင် သတ်မှတ်ရန် လိုအပ်ပါသည်။ သင်၏ model ကို သတ်မှတ်ပြီးနောက်၊ `word_segmenter.train_model()` function ကို သုံး၍ သင်၏ model ကို train နိုင်သည်၊ `word_segmenter.save_model()` ဖြင့် သိမ်းဆည်းနိုင်ပြီး၊ `word_segmenter.test_model_line_by_line()` ဖြင့် စမ်းသပ်နိုင်သည်-
    ```python
    word_segmenter.train_model()
    word_segmenter.save_model()
    word_segmenter.test_model_line_by_line(verbose=True)
    ```
    ဤ repository ကို ဘာသာစကားအသစ်များအတွက် model များ train ခြင်း လုပ်ငန်းစဉ်ကို semi-automatic ဖြစ်စေရန် တီထွင်ထားပါသည်။ အကယ်၍ သင် စိတ်ဝင်စားပါက၊ သင့်တော်သော data set များကို ရှာဖွေရန် (သို့မဟုတ် unsupervised learning option ကို အသုံးပြုရန် ဆုံးဖြတ်ရန်)၊ `word_segmenter.py` နှင့် `constants.py` တွင် ထို data set များကို အသုံးပြုနိုင်စေရန် line အနည်းငယ် ထပ်ထည့်ရန်၊ `hunits` နှင့် `embedding_dim` ၏ တန်ဖိုးများကို ခန့်မှန်းရန် `LSTMBayesianOptimization` class ကို အသုံးပြုရန် (အသေးစိတ်အတွက် [Models Specifications](Models_Specifications_Burmese.md) ကိုကြည့်ပါ)၊ ထို့နောက် အထက်ပါအတိုင်း သင်၏ model များကို train ရန် လိုအပ်ပါသည်။ Grapheme clusters embedding ကို အသုံးပြုရန် ဆုံးဖြတ်ပါက အပို preprocessing အချို့ (see `preproceee.py`) ကိုလည်း ပြုလုပ်ရန် လိုအပ်နိုင်ပါသည်။ ဤကိစ္စနှင့် ပတ်သက်၍ ကျွန်ုပ် ကူညီနိုင်သည်ဟု သင်ထင်ပါက ကျွန်ုပ်ကို ဆက်သွယ်ပါ။

### Model တည်ဆောက်ပုံ
ပုံ ၁ တွင် ကျွန်ုပ်တို့၏ bi-directional model တည်ဆောက်ပုံကို ဖော်ပြထားသည်။ အောက်တွင် layer အမျိုးမျိုးကို ရှင်းပြထားပါသည်-

*   **Input Layer**: Input layer တွင် ကျွန်ုပ်တို့ segment လုပ်လိုသော စာလုံးများ သို့မဟုတ် character များ၏ sequence တစ်ခု ပါဝင်သည်။ ပိုမိုတိကျစွာ ပြောရလျှင်၊ string တစ်ခုကို ကြည့်သည့်အခါ ၎င်းကို code point များ သို့မဟုတ် [extended grapheme clusters](https://unicode.org/reports/tr29/) များ၏ sequence အဖြစ် မြင်နိုင်သည်။ သင်၏ input sequence အတွက် unit များ (grapheme clusters သို့မဟုတ် code points) ကို ရွေးချယ်ခြင်းသည် မတူညီသော model များကို ဖြစ်ပေါ်စေပြီး၊ ၎င်းကို ဂရုတစိုက် ရွေးချယ်သင့်သည်။ ဤ repository ရှိ code သည် ဤ option နှစ်ခုလုံးကို ထောက်ပံ့ပေးသည်။

*   **Embedding Layer**: Embedding layer တွင်၊ input line ၏ unit တစ်ခုစီ (grapheme cluster သို့မဟုတ် code point) ကို model ၏ ကျန်အပိုင်းများက အသုံးပြုနိုင်ရန် numerical vector တစ်ခုဖြင့် ကိုယ်စားပြုသည်။ Embedding ကို ရွေးချယ်ခြင်းသည် model အရွယ်အစားနှင့် performance ကို သိသိသာသာ သက်ရောက်မှုရှိနိုင်သည်။ Embedding vector တစ်ခုစီ၏ အရှည်ကို ဤ document ၏ ကျန်အပိုင်းများတွင် *embedding size* ဖြင့် ဖော်ပြသည်။ ဤ repository တွင် embedding အမျိုးအစား သုံးမျိုးကို အကောင်အထည်ဖော်ထားသည်-
    *   **grapheme clusters to vectors**: ဤချဉ်းကပ်မှုတွင်၊ grapheme cluster တစ်ခုစီကို vector တစ်ခုတည်းသို့ map လုပ်သည်။ ဤ vector များကို model training ပြုလုပ်စဉ်တွင် သင်ယူပြီး နောက်ပိုင်းတွင် evaluation အတွက် အသုံးပြုရန် သိမ်းဆည်းထားရန် လိုအပ်သည်။ ဖြစ်နိုင်သော grapheme cluster များ၏ set သည် သီအိုရီအရ အဆုံးမရှိသောကြောင့်၊ ဖြစ်နိုင်သော grapheme cluster တစ်ခုစီအတွက် vector တစ်ခုကို သိမ်းဆည်းနိုင်ခြင်းမရှိပါ။ ထို့ကြောင့်၊ စာသားများတွင် အမှန်တကယ် ဖြစ်ပေါ်သော grapheme cluster အားလုံးကို ထုတ်ယူရန် corpus ကြီးများကို အသုံးပြုသည်။ ထို့နောက် ဤ grapheme cluster များကို corpus များရှိ သူတို့၏ frequency အပေါ် အခြေခံ၍ စီပြီး၊ စာသား၏ ၉၉% ကို လွှမ်းခြုံသော grapheme cluster များအတွက် vector တစ်ခုနှင့် အခြား grapheme cluster များအတွက် vector တစ်ခုကို သိမ်းဆည်းသည်။ ဤချဉ်းကပ်မှုကို အသုံးပြု၍၊ ထိုင်းနှင့် မြန်မာအတွက် grapheme cluster vector ၃၅၀ ခန့်ကို သိမ်းဆည်းရန် လိုအပ်သည်။
    *   **Generalized encoding vectors**: ဤချဉ်းကပ်မှုတွင်၊ code point တစ်ခုစီကို training ပြုလုပ်စဉ်တွင် သင်ယူသော vector တစ်ခုသို့ map လုပ်ပြီး၊ grapheme cluster အတွက် တွက်ချက်ထားသော vector သည် ၎င်းတွင်ပါဝင်သော code point များနှင့် သက်ဆိုင်သော vector များ၏ ပျမ်းမျှဖြစ်သည်။ ဘာသာစကားတစ်ခုရှိ code point အရေအတွက်သည် ပုံသေဖြစ်ပြီး grapheme cluster အရေအတွက်ထက် သိသိသာသာ နည်းပါးသောကြောင့်၊ ဤချဉ်းကပ်မှုကို အသုံးပြု၍ embedding matrix သည် ပိုမိုသေးငယ်သော အရွယ်အစားရှိမည်ဖြစ်သည်။ သို့သော်၊ ဤချဉ်းကပ်မှုသည် ပုံမှန်အားဖြင့် model တွင် hidden unit နှင့် embedding size အရေအတွက် ပိုမိုများပြားရန် လိုအပ်သည်။ ဤချဉ်းကပ်မှု၏ မတူညီသော ပုံစံများ ([Embedding Discussion](Embeddings_Discussion_Burmese.md) တွင် ရှင်းပြထားသည်) သည် ကိန်းဂဏန်းများကဲ့သို့ အလားတူ ပြုမူသည်ဟု ကျွန်ုပ်တို့ ယုံကြည်သော code point များ အုပ်စုတစ်ခုအတွက် vector တစ်ခုရှိခြင်း ဟူသော idea ကို ဗဟိုပြုသည်။
    *   **code points to vectors**: ဤချဉ်းကပ်မှုတွင်၊ code point တစ်ခုစီကို vector တစ်ခုတည်းသို့ map လုပ်ပြီး ဤ vector များကို model ၏ ကျန်အပိုင်းများတွင် တိုက်ရိုက် အသုံးပြုသည်။ ထို့ကြောင့်၊ ယခင် method နှစ်ခုနှင့် မတူဘဲ၊ ဤနေရာတွင် ဝါကျတစ်ကြောင်း၏ အသေးငယ်ဆုံး အပိုင်းသည် grapheme cluster များအစား code point ဖြစ်သည်။ တဖန်၊ ဤချဉ်းကပ်မှုသည် ကျွန်ုပ်တို့ သိမ်းဆည်းရန် လိုအပ်သော vector အရေအတွက်ကို လျော့ကျစေသော်လည်း၊ hidden unit အရေအတွက်နှင့် embedding size ကို မကြာခဏ တိုးပွားစေသည်။
*   **Forward/Backward LSTM Layers**: Embedding layer ၏ output ကို forward နှင့် backward LSTM layer များသို့ ပေးပို့သည်။ LSTM ၏ cell တစ်ခုစီရှိ hidden unit အရေအတွက်ကို *hunits* ဖြင့် ပြသသည်။

*   **Output Layer**: ဤနေရာတွင်၊ forward နှင့် backward LSTM layer များ၏ output များကို ပေါင်းစပ်ပြီး *softmax* activation function ဖြင့် dense layer တစ်ခုသို့ ပေးပို့ကာ grapheme cluster တစ်ခုစီအတွက် အရှည် လေးခုရှိသော vector တစ်ခုကို ပြုလုပ်သည်။ vector တစ်ခုစီရှိ တန်ဖိုးများသည် ၁ သို့ ပေါင်းပြီး *BIES* ၏ probabilities များဖြစ်သည်၊ ထိုနေရာတွင်-
    *   *B* သည် စကားလုံး၏ အစ (beginning) ဖြစ်သည်။
    *   *I* သည် စကားလုံး၏ အတွင်း (inside) ဖြစ်သည်။
    *   *E* သည် စကားလုံး၏ အဆုံး (end) ဖြစ်သည်။
    *   *S* သည် စကားလုံးတစ်လုံးတည်းကို ဖွဲ့စည်းသော တစ်ခုတည်းသော grapheme cluster (single) ဖြစ်သည်။

*   **Dropout Layers**: ကျွန်ုပ်တို့၏ model တွင် dropout layer နှစ်ခုရှိသည်; တစ်ခုသည် embedding layer ပြီးနောက် ချက်ချင်းနှင့် တစ်ခုသည် output layer မတိုင်မီ ဖြစ်သည်။


![Figure 1. The model structure for a bi-directional LSTM.](../../Figures/model_structure.png)

### Model ၏ hyperparameters များကို ခန့်မှန်းခြင်း
Model တွင် ၎င်းကို အသုံးမပြုမီ ခန့်မှန်းရန် လိုအပ်သော hyperparameter များစွာရှိသည်။ မတူညီသော hyper-parameter များထဲတွင်၊ model အရွယ်အစားနှင့် performance ကို ပိုမိုသိသိသာသာ သက်ရောက်မှုရှိသော နှစ်ခုမှာ *hunits* နှင့် *embedding size* ဖြစ်သည်။ ကျွန်ုပ်တို့သည် *learning rate*, *batch size*, နှင့် *dropout rate* ကဲ့သို့သော ဤနှစ်ခုမှလွဲ၍ hyper-parameter အားလုံးကို ဆုံးဖြတ်ရန် stepwise grid-search ကို အသုံးပြုသည်။ *hunits* နှင့် *embedding size* အတွက် ကျွန်ုပ်တို့သည် [Bayesian optimization](https://github.com/fmfn/BayesianOptimization) ကို အသုံးပြုသည်၊ ၎င်းသည် တွက်ချက်မှုအရ ပိုမိုစျေးကြီးသော်လည်း ဤ parameter များ၏ ပိုမိုကောင်းမွန်သော ခန့်မှန်းမှုကို အာမခံသည်။

### Data sets
အချို့သော ဘာသာစကားများအတွက်၊ learning-based model များကို train ရန် အသုံးပြုနိုင်သော manually annotated data set များ ရှိပါသည်။ သို့သော်၊ အချို့သော အခြား ဘာသာစကားများအတွက်၊ ထိုကဲ့သို့သော data set များ မရှိပါ။ ကျွန်ုပ်တို့သည် scenario နှစ်ခုလုံးတွင် ကျွန်ုပ်တို့၏ model ကို train နိုင်စေသော framework တစ်ခုကို တီထွင်သည်။ ဤ framework တွင် (ပုံ ၂ တွင် ပြထားသည်)၊ အကယ်၍ manually segmented data set တစ်ခု ရှိပါက၊ ကျွန်ုပ်တို့သည် ၎င်းကို ကျွန်ုပ်တို့၏ model ကို တိုက်ရိုက် train ရန် အသုံးပြုသည် (supervised learning)။ သို့မဟုတ်၊ အကယ်၍ ထိုကဲ့သို့သော data set မရှိပါက (unsupervised learning)၊ ကျွန်ုပ်တို့သည် pseudo segmented data ကို ထုတ်လုပ်ရန် လက်ရှိ ICU algorithm ကဲ့သို့သော ရှိပြီးသား algorithm များထဲမှ တစ်ခုကို အသုံးပြုပြီး၊ ထို့နောက် ၎င်းကို ကျွန်ုပ်တို့၏ model ကို train ရန် အသုံးပြုသည်။ ကျွန်ုပ်တို့သည် ICU ကို အထူးသဖြင့် အသုံးပြုသည်၊ အဘယ်ကြောင့်ဆိုသော် ၎င်းသည် ဘာသာစကားအားလုံးနီးပါးအတွက် word segmentation ကို ထောက်ပံ့ပေးပြီး၊ ပေါ့ပါးသည်၊ မြန်ဆန်သည်၊ နှင့် လက်ခံနိုင်သော accuracy ရှိသည်။ သို့သော်၊ ပိုမိုကောင်းမွန်သော word segmentation algorithm များရှိသော အချို့သော ဘာသာစကားများအတွက် ICU ကို အစားထိုးနိုင်သည်။ ကျွန်ုပ်တို့၏ သုံးသပ်ချက်အရ segmented data set မရှိသည့်အခါ၊ ကျွန်ုပ်တို့၏ algorithm သည် ICU လုပ်ဆောင်သည်ကို သင်ယူနိုင်စွမ်းရှိပြီး၊ အချို့ကိစ္စများတွင် ၎င်းသည် ICU ကို ကျော်လွန်နိုင်သည်။ အောက်တွင် ထိုင်းနှင့် မြန်မာအတွက် model များကို train ရန်နှင့် စမ်းသပ်ရန် အသုံးပြုသော data set များကို ရှင်းပြထားပါသည်-

*   **Thai**: ကျွန်ုပ်တို့သည် ကျွန်ုပ်တို့၏ model ကို train ရန် [NECTEC BEST data set](https://thailang.nectec.or.th/downloadcenter/index4c74.html?option=com_docman&task=cat_view&gid=42&Itemid=61) ကို အသုံးပြုသည်။ ဤ dataset ရှိ text file များသည် UTF-8 encoding ကို အသုံးပြုပြီး manually segmented လုပ်ထားသည်။ ဤ data set တွင် text အမျိုးအစား လေးမျိုးရှိသည်- novel, news, encyclopedia, နှင့် article။ model ကို စမ်းသပ်ရန်၊ ကျွန်ုပ်တို့သည် NECTEC BEST data set နှင့် Google SAFT data နှစ်ခုလုံးကို အသုံးပြုသည်။
*   **Burmese**: မြန်မာအတွက်၊ ကျွန်ုပ်တို့သည် unsegmented text များကို စုဆောင်းရန် [Google corpus crawler](https://github.com/google/corpuscrawler) ကို အသုံးပြုပြီး၊ ထို့နောက် training အတွက် အသုံးပြုရန် pseudo segmented data set တစ်ခုကို ထုတ်လုပ်ရန် ICU ကို အသုံးပြုသည်။ စမ်းသပ်ရန်အတွက်၊ ကျွန်ုပ်တို့သည် pseudo segmented text များနှင့် Google SAFT data နှစ်ခုလုံးကို အသုံးပြုသည်။

![Figure 2. The framework for training and testing the model.](../../Figures/framework.png)

### Performance အကျဉ်းချုပ်
Trained model များ၏ set နှစ်ခုရှိသည်၊ တစ်ခုသည် language-specific script ကို အသုံးပြု၍ train ထားသော model များ (သူတို့၏ အမည်တွင် `exclusive` ပါသော model များ) ဖြစ်ပြီး၊ space, mark, နှင့် Latin စာလုံးများ အပါအဝင် အခြား character အားလုံးကို data မှ ဖယ်ထုတ်ထားသည်။ ၎င်းသည် model ကို ပိုမိုသေးငယ်သော ဝါကျများဖြင့် train ရန် တွန်းအားပေးပြီး ၎င်း၏ accuracy ကို လျော့ကျစေနိုင်သည်။ သို့သော်၊ ဤ model များသည် ICU4C word segmenter ၏ တည်ဆောက်ပုံနှင့် လုံးဝ လိုက်ဖက်ညီပြီး၊ ထိုင်းနှင့် မြန်မာအတွက် language engine များကို တိုက်ရိုက် အစားထိုးနိုင်သည်။ ဒုတိယ set ၏ model များကို standard data set များ (space, mark, Latin စာလုံးများ ပါဝင်သော) ကို အသုံးပြု၍ train ထားပြီး ပိုမိုကောင်းမွန်သော accuracy များကို ပေးသည်။ ဤ model များကို ICU4X တွင် အသုံးပြုနိုင်ပြီး၊ ICU4C တွင်လည်း ၎င်း၏ လက်ရှိ တည်ဆောက်ပုံကို အပြောင်းအလဲ အချို့ပြုလုပ်ပါက အသုံးပြုနိုင်သည်။ အောက်တွင် ပထမ set ၏ model များ၏ performance ကို ဖော်ပြပြီး ရှိပြီးသား algorithm များနှင့် နှိုင်းယှဉ်ထားပါသည်-

*   **Thai**: အောက်ပါဇယားသည် ကျွန်ုပ်တို့၏ algorithm ၏ performance ကို state of the art algorithm [Deepcut](https://github.com/rkcosmos/deepcut) နှင့် လက်ရှိ ICU algorithm တို့နှင့်အတူ အကျဉ်းချုပ်ဖော်ပြသည်။ ကျွန်ုပ်တို့၏ algorithm ၏ မတူညီသော version များရှိသည်၊ LSTM model 7 နှင့် LSTM model 5 တို့သည် အသီးသီး အတိကျဆုံးနှင့် အချွေတာဆုံး model များဖြစ်ပြီး၊ LSTM model 4 သည် ဤနှစ်ခုကြားတွင် ရှိပြီး မြင့်မားသော accuracy ကို ပေးစွမ်းသော်လည်း data အရွယ်အစားမှာ သေးငယ်သည်။ ဤဇယားအရ၊ BEST data ကို evaluation အတွက် အသုံးပြုသည့်အခါ၊ **LSTM model အားလုံးသည် F1-score အပေါ် အခြေခံ၍ ICU ကို ကျော်လွန်သည်**၊ model 4 နှင့် 7 တို့သည် သိသာသော margin ဖြင့် ပြုလုပ်သည်။ SAFT data ကို အသုံးပြုသည့်အခါ၊ ကျွန်ုပ်တို့၏ model များ၏ performance တွင် သိသာသော ကျဆင်းမှုကို တွေ့ရသည်၊ ၎င်းသည် အဓိကအားဖြင့် BEST နှင့် SAFT data များတွင် မတူညီသော segmentation rule များကြောင့် ဖြစ်သည် (ဥပမာ 2020 ကို BEST တွင် |2020| အဖြစ် နှင့် SAFT တွင် |2|0|2|0| အဖြစ် segment လုပ်သည်)။ သို့သော်၊ LSTM model များသည် ICU ကို ကျော်လွန်နေဆဲဖြစ်သည်။ **Data အရွယ်အစားအရ၊ LSTM model 4, 5, နှင့် 7 တို့သည် အသီးသီး 79%, 92%, နှင့် 32% လျော့ချမှု ပြသသည်**။ Deepcut သည် အကြီးဆုံး model ဖြစ်ပြီး ၎င်း၏ application များကို industry တွင် ကန့်သတ်ထားသည်။ ၎င်းသည် BEST data ကို အသုံးပြုသည့်အခါ accuracy အရ အခြား method အားလုံးကို သိသာသော margin ဖြင့် ကျော်လွန်သည်။ သို့သော်၊ SAFT data အတွက်၊ Deepcut ကို train ရန် အသုံးမပြုသော data ဖြစ်သောကြောင့်၊ ဤ margin သည် သိသိသာသာ ကျဆင်းသည်။

| Algorithm | BIES accuracy (BEST) | F1-score (BEST) | BIES accuracy (SAFT) | F1-score (SAFT) | Model size |
| :---: | :----: | :---: | :----: | :---: | :---: |
| LSTM model 4 | 94.5 | 89.9 | 90.8 | 82.8 | 27 KB |
| LSTM model 5 | 92.6 | 86.6 | 88.9 | 79.6 | 10 KB |
| LSTM model 7 | 95.7 | 91.9 | 92 | 84.9 | 86 KB |
| Deepcut | 97.8 | 95.7 | 92.5 | 86 | 2.2 MB |
| ICU | 93 | 86.4 | 90.3 | 81.9 | 126 KB |

*   **Burmese**:
    အောက်ပါဇယားသည် မြန်မာအတွက် ကျွန်ုပ်တို့၏ algorithm နှင့် လက်ရှိ ICU algorithm ၏ performance ကို အကျဉ်းချုပ်ဖော်ပြသည်။ တဖန်၊ ကျွန်ုပ်တို့၏ LSTM model များ၏ မတူညီသော version များရှိသည်၊ LSTM model 7 နှင့် LSTM model 5 တို့သည် အသီးသီး အတိကျဆုံးနှင့် အချွေတာဆုံး model များဖြစ်သည်။ ဤဇယားအရ၊ LSTM model များသည် ICU algorithm လုပ်ဆောင်သည်ကို ကောင်းစွာ အတုခိုးရန် သင်ယူသည်။ **ဥပမာအားဖြင့်၊ SAFT data တွင်၊ ICU နှင့် စပ်လျဉ်း၍ relative error သည် model 7 အတွက် 1% (93.1/92.4) ထက် နည်းပါးပြီး model 4 နှင့် 5 အတွက် 2% ထက် နည်းပါးသည်။ Data အရွယ်အစားအရ၊ LSTM model 4, 5, နှင့် 7 တို့သည် အသီးသီး 88%, 94%, နှင့် 51% လျော့ချမှု ပြသသည်**။

| Algorithm | BIES accuracy (ICU segmented) | F1-score (ICU segmented) | BIES accuracy (SAFT) | F1-score (SAFT) | Model size |
| :---: | :----: | :---: | :---: | :---: | :---: |
| LSTM model 4 | 94.7 | 92.9 | 91.7 | 90.5 | 30 KB |
| LSTM model 5 | 93.4 | 91.1 | 91.4 | 90.1 | 15 KB |
| LSTM model 7 | 96.2 | 94.9 | 92.3 | 91.1 | 125 KB |
| ICU | 100 | 100 | 93.1 | 92.4 | 254 KB |

ဤ project ကို တိုးတက်စေရန် လမ်းကြောင်းများစွာရှိသည်။ ကျွန်ုပ်တို့၏ idea အချို့အတွက် [Future Works](https://github.com/SahandFarhoodi/word_segmentation/blob/work/Future%20Works.md) ကို ဖတ်ရှုပြီး၊ အကယ်၍ သင့်တွင် idea တစ်ခုခုရှိပါက [ကျွန်ုပ်](http://math.bu.edu/people/sahand/)ကို ဆက်သွယ်ပါ။

### Copyright & Licenses

Copyright © 2020-2024 Unicode, Inc. Unicode နှင့် Unicode Logo တို့သည် United States နှင့် အခြားနိုင်ငံများတွင် Unicode, Inc. ၏ မှတ်ပုံတင်ထားသော ကုန်အမှတ်တံဆိပ်များ ဖြစ်သည်။

ဤ project သို့ contribute လုပ်ရန် CLA လိုအပ်သည် - ပိုမိုသောအချက်အလက်များအတွက် [CONTRIBUTING.md](https://github.com/unicode-org/.github/blob/main/.github/CONTRIBUTING.md) ဖိုင်ကို ဖတ်ရှုပါ (သို့မဟုတ် Pull Request စတင်ပါ)။

ဤ repository ၏ အကြောင်းအရာများကို Unicode [Terms of Use](https://www.unicode.org/copyright.html) ဖြင့် အုပ်ချုပ်ပြီး [LICENSE](./LICENSE) အောက်တွင် ထုတ်ပြန်သည်။
