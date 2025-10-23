## Embedding System အမျိုးမျိုးအတွက် Error သုံးသပ်ချက်
ရေးသားသူ: Sahand Farhoodi (sahandfr@gmail.com, sahand.farhoodi93@gmail.com)

ဤ project တွင်၊ ကျွန်ုပ်တို့သည် word segmentation အတွက် bi-directional LSTM model တစ်ခုကို တီထွင်ထားပါသည်။ ယခုအချိန်တွင်၊ ဤ model ကို Thai (ထိုင်း) နှင့် Burmese (မြန်မာ) ဘာသာစကားများအတွက် အကောင်အထည်ဖော်ထားပါသည်။

### နောက်ခံ
ဤ document တွင်၊ **ထိုင်း** သို့မဟုတ် **မြန်မာ** ဘာသာစကားဖြင့် ပေးထားသော ဝါကျတစ်ကြောင်း၏ word boundaries များကို ခန့်မှန်းပေးသော [bi-directional LSTM word segmentation algorithm](https://github.com/SahandFarhoodi/word_segmentation) အတွက် embedding system အမျိုးမျိုးကို စူးစမ်းလေ့လာပါမည်။ ဤ model ၏ ပထမဆုံး layer သည် embedding layer ဖြစ်ပြီး၊ LSTM model ၏ input ဖြစ်သော character သို့မဟုတ် စာလုံးများ၏ sequence ကို numerical vector များ၏ sequence သို့ map လုပ်ပါသည်။ Embedding layer အတွက် [ဤ document](https://docs.google.com/document/d/1KXnTvrgISYUplOk1NRbQbJssueeXa8k1Vu8YApMud4k/edit#heading=h.bmtbd2h7j5nt) တွင် အသေးစိတ် ရှင်းပြထားသော option များစွာရှိပါသည်။ ဤနေရာတွင်၊ ကျွန်ုပ်တို့သည် ဤ embedding system များထဲမှ သုံးခုကို သုံးသပ်ပြီး ၎င်းတို့အတွက် word segmentation algorithm ကို အကောင်အထည်ဖော်ကာ model accuracy နှင့် model size အပေါ် အခြေခံ၍ နှိုင်းယှဉ်ပါမည်။ ဤ မတူညီသော embedding system များအပေါ် အခြေခံ၍ error သုံးသပ်ချက်များ ပြုလုပ်ရန်လည်း အချိန်အနည်းငယ် အသုံးပြုပါမည်။

Embedding system သည် ကျွန်ုပ်တို့၏ model က training အတွက် အသုံးပြုရန် စဉ်းစားသော ဝါကျ၏ unit များအပေါ် များစွာ မူတည်ပါသည်။ ရွေးချယ်စရာ နှစ်ခုရှိသည်-
*   **Unicode code points**: ၎င်းတို့ကို ဝါကျ သို့မဟုတ် စကားလုံးတစ်လုံး၏ အသေးငယ်ဆုံး အစိတ်အပိုင်းအဖြစ် မကြာခဏ သတ်မှတ်သည်။ တရားဝင် definition အတွက် [ဤ link](https://en.wikipedia.org/wiki/Code_point) ကို ကြည့်ရှုပါ။
*   **Grapheme clusters**: grapheme cluster တစ်ခုစီတွင် code point တစ်ခု သို့မဟုတ် တစ်ခုထက်ပို၍ ပါဝင်နိုင်သည်။ ဤ grapheme cluster အမျိုးအစားနှစ်ခုကို single-code-point grapheme cluster နှင့် multi-code-point grapheme cluster ဟု ခွဲခြားနိုင်သည်။ Grapheme cluster များကို word segmentation algorithm များအတွက် အခြေခံ unit အဖြစ် သတ်မှတ်ရသည့် အကြောင်းရင်းမှာ word boundary သည် grapheme cluster တစ်ခု၏ အလယ်တွင် ဘယ်တော့မှ မဖြစ်သင့်သောကြောင့် ဖြစ်သည်။

ပေးထားသော ဘာသာစကားတစ်ခု၊ ဥပမာ ထိုင်းအတွက်၊ ဤ document တွင် စူးစမ်းလေ့လာထားသော embedding system သုံးခုမှာ-

*   **grapheme clusters vectors** (graph_clust): ဤ embedding system တွင်၊ ပထမဦးစွာ ပေးထားသော ဘာသာစကားရှိ စာသား၏ ၉၉% ကို လွှမ်းခြုံသော grapheme cluster များ၏ set တစ်ခုကို ရှာဖွေသည်။ ဤ set ကို ရရှိရန် ပေးထားသော ဘာသာစကားရှိ corpus ကြီးများကို အသုံးပြုသည်။ ထို့နောက်၊ ဤ set ရှိ grapheme cluster တစ်ခုစီကို ပုံသေ အရှည်ရှိသော vector တစ်ခုတည်းဖြင့် ကိုယ်စားပြုသည်။ အခြား grapheme cluster အားလုံးကို တူညီသော အရှည်ရှိသော shared vector တစ်ခုဖြင့် ကိုယ်စားပြုသည်။ ထိုင်းနှင့် မြန်မာအတွက်၊ ၎င်းသည် ကျွန်ုပ်တို့အား vector ~350 ပါသော embedding matrix တစ်ခုကို ပေးသည်။ ဤ vector များကို LSTM model ၏ training ပြုလုပ်စဉ်တွင် သင်ယူသည်။

    ![Figure 1. Grapheme clusters embedding.](../../Figures/Graphclust_embedding.png)

*   **generalized vectors**: ဤ embedding system တွင်၊ ပေးထားသော ဘာသာစကားရှိ Unicode code point (သို့မဟုတ် code point အုပ်စု) တစ်ခုစီကို ပုံသေ အရှည်ရှိသော vector တစ်ခုဖြင့် ကိုယ်စားပြုသည်။ ထို့နောက်၊ grapheme cluster တစ်ခုစီကို ၎င်းတွင်ပါဝင်သော code point များနှင့် သက်ဆိုင်သော vector များ၏ ပျမ်းမျှဖြင့် ကိုယ်စားပြုသည်။

    ![Figure 2. Generalized vectors embedding.](../../Figures/Genvec_embedding.png)

    ဤ embedding system တွင် version အမျိုးမျိုးရှိသည်-
    *   **Buckets 1, 2** (genvec_12): ဤ version တွင်၊ ပေးထားသော ဘာသာစကားရှိ type 1 (letters) သို့မဟုတ် 2 (marks) ရှိသော code point တစ်ခုစီကို သီးခြား vector တစ်ခုဖြင့် ကိုယ်စားပြုသည်။ အခြား code point များကို အောက်ပါအတိုင်း အုပ်စုဖွဲ့ပြီး၊ အုပ်စုတစ်ခုစီ၏ element များသည် vector တစ်ခုတည်းကို share လုပ်သည်-
        *   အထက်တွင် ကိုယ်စားမပြုထားသော အခြား စာလုံးများအားလုံး
        *   အထက်တွင် ကိုယ်စားမပြုထားသော အခြား ဂဏန်းများအားလုံး
        *   အထက်တွင် ကိုယ်စားမပြုထားသော mark, punctuation, နှင့် symbol အားလုံး
        *   type 4 (separators) နှင့် 7 (others) ရှိ code point အားလုံး

        ဤ version တွင်၊ ကျွန်ုပ်တို့တွင် ရှိမည်-
        *   မြန်မာအတွက် embedding vector 136 ခု
        *   ထိုင်းအတွက် embedding vector 77 ခု

    *   **Buckets 1, 2, 3** (genvec_123): "Bucket 1,2" version နှင့် တူညီသော်လည်း၊ ယခုအခါ ပေးထားသော ဘာသာစကားရှိ type 3 (digits) ရှိသော code point တစ်ခုစီကိုလည်း သီးခြား vector တစ်ခုဖြင့် ကိုယ်စားပြုသည်။ ဤ version တွင်၊ ကျွန်ုပ်တို့တွင် ရှိမည်-
        *   မြန်မာအတွက် embedding vector 156 ခု
        *   ထိုင်းအတွက် embedding vector 87 ခု

    *   **Buckets 1, 2, digit 0** (genvec_12d0): "Bucket 1,2" version နှင့် တူညီသော်လည်း၊ ယခုအခါ ပေးထားသော ဘာသာစကားရှိ digit 0 ကို ကိုယ်စားပြုသော code point များကို vector အသစ်တစ်ခုဖြင့် ကိုယ်စားပြုသည်။ ဤ version တွင်၊ ကျွန်ုပ်တို့တွင် ရှိမည်-
        *   မြန်မာအတွက် embedding vector 138 ခု
        *   ထိုင်းအတွက် embedding vector 78 ခု

    *   **Buckets 1, 2, 5** (genvec_125): "Bucket 1,2" version နှင့် တူညီသော်လည်း၊ ယခုအခါ ပေးထားသော ဘာသာစကားရှိ type 5 (punctuations) ရှိသော code point တစ်ခုစီကိုလည်း သီးခြား vector တစ်ခုဖြင့် ကိုယ်စားပြုသည်။ ဤ version တွင်၊ ကျွန်ုပ်တို့တွင် ရှိမည်-
        *   မြန်မာအတွက် embedding vector 142 ခု
        *   ထိုင်းအတွက် embedding vector 80 ခု

    *   **Buckets 1, 2, 3, 5** (genvec_1235): "Bucket 1,2" version နှင့် တူညီသော်လည်း၊ ယခုအခါ ပေးထားသော ဘာသာစကားရှိ type 3 (digits) သို့မဟုတ် type 5 (punctuations) ရှိသော code point တစ်ခုစီကိုလည်း သီးခြား vector တစ်ခုဖြင့် ကိုယ်စားပြုသည်။ ဤ version တွင်၊ ကျွန်ုပ်တို့တွင် ရှိမည်-
        *   မြန်မာအတွက် embedding vector 162 ခု
        *   ထိုင်းအတွက် embedding vector 90 ခု

*   **Code points**: ဤ embedding system တွင်၊ ပေးထားသော ဘာသာစကားရှိ Unicode code point တစ်ခုစီကို vector တစ်ခုတည်းဖြင့် ကိုယ်စားပြုသည်။ ဤတည်ဆောက်ပုံသည် model အား grapheme cluster တစ်ခုအတွင်းတွင် word boundary များ ထားရှိနိုင်စေရန် အလားအလာရှိသည်။ ထို့ကြောင့်၊ ဤ word boundary အမျိုးအစားကို စစ်ဆေးပြီး ပြင်ဆင်ပေးသော အပို normalizer algorithm တစ်ခု လိုအပ်သည်။

    ![Figure 3. Code points embedding.](../../Figures/Codepoints_embedding.png)

### Error သုံးသပ်ချက်
ဤအပိုင်းတွင် ဖော်ပြထားသော model အားလုံးကို တူညီသော training data set များဖြင့် train ထားပြီး၊ တူညီသော epoch အရေအတွက်နှင့် batch size များကို အသုံးပြုထားသည်။ အရေးကြီးသော hyperparameter နှစ်ခုဖြစ်သော hidden unit အရေအတွက် (hunits) နှင့် embedding dimension ၏ တန်ဖိုးကို embedding system တစ်ခုစီအတွက် Bayesian optimization ကို အသုံးပြု၍ သီးခြား တွက်ချက်ထားသည်။ ဤ hyperparameter နှစ်ခုအတွက် တွက်ချက်ထားသော တန်ဖိုးမှာ အောက်ပါအတိုင်းဖြစ်သည်-

| Embedding | embedding dimension | hunits |
| :---: | :----: | :---: |
| Thai graph_clust | 16 | 23 |
| Thai codepoints | 40 | 27 |
| Thai genvec | 22 | 40 |
| Thai genvec light | 22 | 20 |
| Burmese graph_clust | 28 | 14 |
| Burmese genvec | 33 | 20 |

#### Grapheme Clusters vs. Generalized Vectors
ဤနေရာတွင်၊ grapheme cluster နှင့် generalized vector embedding system များအကြား ခြားနားချက်ကို စူးစမ်းလေ့လာပါမည်။ အောက်ပါဇယားသည် **ထိုင်း** အတွက် fit လုပ်ထားသော LSTM model များ၏ accuracy နှင့် model size ကို ပြသသည်-

| Embedding | BIES accuracy | F1-Score | Model size |
| :---: | :----: | :----: | :---: |
| graph_clust | 92 | 85.3 | 27 KB |
| graph_clust_light | 91.9 | 85.2 | 22 KB |
| genvec_12 | 92.2 | 85.5 | 47 KB |
| genvec_123 | 92.3 | 85.7 | 47 KB |
| genvec_123_light | 91.9 | 85 | 20 KB |
| genvec_12d0 | 92.2 | 85.5 | 47 KB |
| genvec_125 | 92.1 | 85.3 | 47 KB |
| genvec_1235 | 92.1 | 85.5 | 47 KB |

အောက်ပါဇယားသည် **မြန်မာ** အတွက် fit လုပ်ထားသော LSTM model များ၏ accuracy နှင့် model size ကို ပြသသည်-

| Embedding | BIES accuracy | F1-Score | Model size |
| :---: | :----: | :----: | :---: |
| graph_clust | 91.7 | 90.4 | 30 KB |
| genvec_12 | 91.2 | 89.9 | 28 KB |
| genvec_123 | 91 | 89.6 | 30 KB |
| genvec_12d0 | 90.7 | 89.3 | 28 KB |
| genvec_125 | 91.1 | 89.7 | 28 KB |
| genvec_1235 | 91.5 | 90.1 | 29 KB |

ဤဇယားများအရ၊ ထိုင်းအတွက်၊ generalized vector များဖြင့် ရရှိနိုင်သမျှသည် grapheme cluster များဖြင့်လည်း ရရှိနိုင်သည်ဟု ထင်ရသည် (`graph_clust_light` ကို `genvec_123_light` နှင့် `graph_clust` ကို `genvec_123` နှင့် နှိုင်းယှဉ်ပါ)။ မြန်မာအတွက်၊ grapheme cluster များကို အသုံးပြုခြင်းသည် ပိုကောင်းသော ရွေးချယ်မှုဖြစ်သည်ဟု ထင်ရသည်; ၎င်းသည် data size ပိုမိုသေးငယ်ပြီး accuracy အရ ပိုမိုကောင်းမွန်သော performance ကို ပြသသည်။ သို့သော်၊ ကျွန်ုပ်တို့၏ သုံးသပ်ချက်အရ generalized vector ချဉ်းကပ်မှုသည်လည်း အလွန်အသုံးဝင်ပြီး၊ grapheme cluster များ အသုံးပြု၍ ရရှိနိုင်သော model accuracy နှင့် အလွန်နီးစပ်သော accuracy ကို ပေးနိုင်သည်ကို ပြသသည်။

#### Grapheme Clusters vs. Code Points

ဤနေရာတွင်၊ grapheme cluster နှင့် code point embedding system များအကြား ခြားနားချက်ကို စူးစမ်းလေ့လာပါမည်။ အောက်ပါဇယားနှစ်ခုသည် ထိုင်းနှင့် မြန်မာအတွက် fit လုပ်ထားသော LSTM model များ၏ accuracy နှင့် model size ကို ပြသသည်။ ဤ embedding system နှစ်ခုအကြား တရားမျှတသော နှိုင်းယှဉ်မှုအတွက်၊ BIES accuracy ကို ကြည့်ခြင်းသည် လွဲမှားစေနိုင်သည်ကို သတိပြုပါ။ ၎င်းသည် code point embedding အတွက် တွက်ချက်ထားသော BIES accuracy ၏ အရှည်သည် grapheme cluster များအတွက် တွက်ချက်ထားသည်ထက် မကြာခဏ ပိုရှည်သောကြောင့်ဖြစ်ပြီး၊ ထို့ကြောင့် BIES accuracy သည် code point embedding များအတွက် ပိုမိုမြင့်မားလေ့ရှိသည်။ ဥပမာအားဖြင့်၊ အောက်ပါဥပမာကို သုံးသပ်ကြည့်ပါ-

*   မှန်ကန်သော segmentation: |ใหญ่ซึ่ง|

*   ကျွန်ုပ်တို့၏ segmentation: |ใหญ่|ซึ่ง|

ဤကိစ္စတွင်၊ grapheme cluster နှင့် code point များအတွက် မှန်ကန်သော BIES sequence၊ ခန့်မှန်းထားသော BIES sequence၊ နှင့် BIES accuracy မှာ-

*   Grapheme cluster:
    *   မှန်ကန်သော segmentation BIES: biiie
    *   ကျွန်ုပ်တို့၏ segmentation BIES: biebe
    *   BIES accuracy = 60

*   Code point:
    *   မှန်ကန်သော segmentation BIES: biiiiiie
    *   ကျွန်ုပ်တို့၏ segmentation BIES: biiebiie
    *   BIES accuracy = 75

ထို့ကြောင့်၊ တူညီသော segmentation အတွက် BIES accuracy ၏ code point version သည် ပိုမိုမြင့်မားသည်။ ဤအကြောင်းကြောင့်၊ ဤအပိုင်းတွင်၊ ကျွန်ုပ်တို့သည် F1-score များကိုသာ report လုပ်သည်။ အောက်ပါဇယားသည် **ထိုင်း** အတွက် fit လုပ်ထားသော LSTM model များ၏ accuracy နှင့် model size ကို ပြသသည်။ ကျွန်ုပ်တို့၏ နှိုင်းယှဉ်မှုသည် grapheme cluster များကို အနည်းငယ် မျက်နှာသာပေးသည်ကို သတိပြုပါ၊ အဘယ်ကြောင့်ဆိုသော် code point အတွက် report လုပ်ထားသော တန်ဖိုးသည် Thai-specific script ကို အသုံးပြု၍ train ထားသော model များအတွက် တွက်ချက်ထားသောကြောင့်ဖြစ်သည်။ ၎င်းသည် မူရင်း data မှ space, mark, နှင့် Latin စာလုံးများအားလုံးကို ဖယ်ထုတ်ခဲ့ရသည်ဟု အဓိပ္ပာယ်ရပြီး၊ ၎င်းသည် ဝါကျများကို ပိုမိုတိုတောင်းစေပြီး model accuracy ကို ဆိုးရွားစွာ သက်ရောက်မှုရှိနိုင်သည်။

| Embedding | F1-Score | Model size |
| :---: | :----: | :---: |
| graph_clust | 88.7 | 28 KB |
| codepoints light | 88.9 | 28 KB |
| codepoints | 90.1 | 36 KB |

အောက်ပါဇယားသည် **မြန်မာ** အတွက် fit လုပ်ထားသော LSTM model များ၏ accuracy နှင့် model size ကို ပြသသည်။ မြန်မာစာသားများတွင် space များစွာ ပါဝင်နိုင်သောကြောင့်၊ Burmese-specific script ကို အသုံးပြု၍ model များကို train ခြင်းသည် accuracy ကို သိသိသာသာ လျော့ကျစေသည်။ ထို့ကြောင့်၊ grapheme cluster နှင့် code point embedding များအကြား တရားမျှတသော နှိုင်းယှဉ်မှု ပြုလုပ်ရန်၊ အောက်ပါဇယားတွင် report လုပ်ထားသော တန်ဖိုးများသည် မူရင်း data set များပေါ်တွင် train ထားသော model များအတွက်ဖြစ်သည်။

| Embedding | F1-Score | Model size |
| :---: | :----: | :---: |
| graph_clust | 92.9 | 30 KB |
| codepoints light | 93 | 20 KB |
| codepoints | 93.4 | 45 KB |

ဤဇယားများအရ၊ တူညီသော size ရှိသော model များအတွက်၊ code point embedding သည် အနည်းငယ် ပိုကောင်းသော accuracy ကို ပေးသည်ကို ကျွန်ုပ်တို့တွေ့ရသည် (ထိုင်းဇယားအတွက် `graph_clust` ကို `codepoint light` နှင့် နှိုင်းယှဉ်ပါ)။ ထို့အပြင်၊ ပိုကြီးသော code point model များကို အသုံးပြုခြင်းဖြင့် ကျွန်ုပ်တို့သည် သိသိသာသာ ပိုကောင်းသော performance ကို ရရှိသည် (ထိုင်းဇယား၏ တတိယတန်း)။ ထို accuracy level ကို grapheme cluster များ အသုံးပြု၍ ရရှိနိုင်ခြင်းမရှိခဲ့ပါ၊ ဤဇယားများတွင် ကျွန်ုပ်တို့ ပြထားသည်ထက် ပိုကြီးသော model များကို train သည့်အခါတွင်ပင်။

#### Case-by-case သုံးသပ်ချက်

အောက်တွင်၊ **ထိုင်း** ဘာသာစကားရှိ ဝါကျအချို့ကို algorithm အမျိုးမျိုးက မည်သို့ segment လုပ်သည်ကို ကြည့်ရန် နမူနာအချို့ကို အသုံးပြုပါမည်-

| Algorithm | Output |
| :---: | :---- |
| Unsegmented | `การเดินทางใน` |
| Manually Segmented | `\|การ\|เดินทาง\|ใน\|` |
| Deepcut | `\|การ\|เดินทาง\|ใน\|` |
| ICU | `\|การ\|เดิน\|ทางใน\|` |
| Grapheme Clusters | `\|การ\|เดิน\|ทาง\|ใน\|` |
| Generalized Vectors | `\|การ\|เดิน\|ทาง\|ใ\|น\|` |
| Code Points | `\|การ\|เดินทาง\|ใน\|` |

**Test Case 3**

| Algorithm | Output |
| :---: | :---- |
| Unsegmented | `นั่งนายกฯต่อสมัยหน้า` |
| Manually Segmented | `\|นั่ง\|นายก\|ฯ\|ต่อ\|สมัย\|หน้า\|` |
| Deepcut | `\|นั่ง\|นายก\|ฯ\|ต่อ\|สมัย\|หน้า\|` |
| ICU | `\|นั่ง\|นา\|ยกฯ\|ต่อ\|สมัย\|หน้า\|` |
| Grapheme Clusters | `\|นั่ง\|นายก\|ฯ\|ต่อ\|สมัย\|หน้า\|` |
| Generalized Vectors | `\|นั่ง\|นายก\|ฯ\|ต่อสมัยหน้า\|` |
| Code Points | `\|นั่ง\|นายก\|ฯ\|ต่อ\|สมัย\|หน้า\|` |

**Test Case 4**

| Algorithm | Output |
| :---: | :---- |
| Unsegmented | `พร้อมจัดตั้ง` |
| Manually Segmented | `\|พร้อม\|จัดตั้ง\|` |
| Deepcut | `\|พร้อม\|จัด\|ตั้ง\|` |
| ICU | `\|พร้อม\|จัด\|ตั้ง\|` |
| Grapheme Clusters | `\|พร้อม\|จัด\|ตั้ง\|` |
| Generalized Vectors | `\|พร้อม\|จัด\|ตั้ง\|` |
| Code Points | `\|พร้อม\|จัดตั้ง\|` |

**Test Case 5**

| Algorithm | Output |
| :---: | :---- |
| Unsegmented | `เพราะดนตรีที่ชอบนั้น` |
| Manually Segmented | `\|เพราะ\|ดนตรี\|ที่\|ชอบ\|นั้น\|` |
| Deepcut | `\|เพราะ\|ดนตรี\|ที่\|ชอบ\|นั้น\|` |
| ICU | `\|เพราะ\|ดนตรี\|ที่\|ชอบ\|นั้น\|` |
| Grapheme Clusters | `\|เพราะ\|ดนตรี\|ที่\|ชอบ\|นั้น\|` |
| Generalized Vectors | `\|เ\|พราะดนตรี\|ที่\|ชอบ\|นั้น\|` |
| Code Points | `\|เพราะ\|ดนตรี\|ที่\|ชอบ\|นั้น\|` |

အောက်တွင်၊ **မြန်မာ** ဘာသာစကားရှိ ဝါကျအချို့ကို algorithm အမျိုးမျိုးက မည်သို့ segment လုပ်သည်ကို ကြည့်ရန် နမူနာအချို့ကို အသုံးပြုပါမည်-

**Test Case 1**

| Algorithm | Output |
| :---: | :---- |
| Unsegmented | `ဖော်ပြထားသည့်` |
| Manually Segmented | `\|ဖော်ပြ\|ထားသည့်\|` |
| ICU | `\|ဖော်ပြ\|ထား\|သည့်\|` |
| Grapheme Clusters | `\|ဖော်\|ပြ\|ထား\|သည့်\| ` |
| Generalized Vectors | `\|ဖော်\|ပြ\|ထား\|သည့်\|` |
| Code Points | `\|ဖော်ပြ\|ထား\|သည့်\` |

**Test Case 2**

| Algorithm | Output |
| :---: | :---- |
| Unsegmented | `အသားအရောင်အားဖြင့်` |
| Manually Segmented | `\|အသားအရောင်\|အားဖြင့်\|` |
| ICU | `\|အသား\|အရောင်\|အားဖြင့်\|` |
| Grapheme Clusters | `\|အသား\|အရောင်\|အား\|ဖြင့်\|` |
| Generalized Vectors | `\|အသား\|အရောင်\|အား\|ဖြင့်\|` |
| Code Points | `\|အသားအရောင်\|အားဖြင့်\|` |

**Test Case 3**

| Algorithm | Output |
| :---: | :---- |
| Unsegmented | `သဘာဝအားဖြင့်` |
| Manually Segmented | `\|သဘာဝ\|အားဖြင့်\|` |
| ICU | `\|သဘာ\|ဝ\|အားဖြင့်\|` |
| Grapheme Clusters | `\|သဘာ\|ဝ\|အား\|ဖြင့်\|` |
| Generalized Vectors | `\|သဘာ\|ဝအား\|ဖြင့်\|` |
| Code Points | `\|သဘာ\|ဝ\|အား\|ဖြင့်\|` |

**Test Case 4**

| Algorithm | Output |
| :---: | :---- |
| Unsegmented | `ထို့ပြင်` |
| Manually Segmented | `\|ထို့ပြင်\|` |
| ICU | `\|ထို့ပြင်\|` |
| Grapheme Clusters | `\|ထို့\|ပြင်\|` |
| Generalized Vectors | `\|ထို့\|ပြင်\|` |
| Code Points | `\|ထို့ပြင်\|` |

**Test Case 5**

| Algorithm | Output |
| :---: | :---- |
| Unsegmented | `နိုင်ငံရေးဆိုင်ရာ` |
| Manually Segmented | `\|နိုင်ငံရေး\|ဆိုင်ရာ\|` |
| ICU | `\|နိုင်ငံရေး\|ဆိုင်ရာ\|` |
| Grapheme Clusters | `\|နိုင်ငံရေး\|ဆိုင်ရာ\|` |
| Generalized Vectors | `\|နိုင်ငံရေး\|ဆိုင်ရာ\|` |
| Code Points | `\|နိုင်ငံရေး\|ဆိုင်ရာ\|` |

### ကောက်ချက်

ဤသုံးသပ်ချက်များအရ၊ မြန်မာနှင့် ထိုင်း နှစ်ဘာသာလုံးအတွက်၊ code point embedding သည် အနှစ်သက်ဆုံး option ဖြစ်သည်ဟု ထင်ရသည်။ ဤ embedding system သည် ပိုမိုကောင်းမွန်သော accuracy များကို ပေးနိုင်ပြီး မကြာခဏ အကောင်အထည်ဖော်ရန် ပိုမိုလွယ်ကူသည်။ ၎င်းသည် training time တွင် grapheme cluster ချဉ်းကပ်မှုထက် preprocessing နည်းပါးစွာ လိုအပ်သည်၊ ထိုနေရာတွင် ကျွန်ုပ်တို့သည် အသုံးအများဆုံး grapheme cluster များကို ဦးစွာ ရှာဖွေရသည်။ ထို့အပြင်၊ evaluation time တွင် ဝါကျတစ်ကြောင်းကို ၎င်း၏ grapheme cluster များအဖြစ် segment လုပ်ရန် မလိုအပ်တော့ပါ။ Code point embedding ၏ နောက်ထပ် အားသာချက်တစ်ခုမှာ grapheme cluster embedding နှင့် မတူဘဲ model တွင် အပို dictionary တစ်ခုခုကို သိမ်းဆည်းရန် မလိုအပ်ခြင်းဖြစ်သည်။

Code point embedding ၏ အားနည်းချက်တစ်ခုမှာ grapheme cluster များအစား code point များကို အသုံးပြုသည့်အခါ ဝါကျများသည် ပိုမိုရှည်လျားသောကြောင့်၊ ဤ embedding type သည် တူညီသော context ပမာဏကို သင်ယူရန် model များတွင် hidden unit အရေအတွက် ပိုမိုများပြားရန် လိုအပ်နိုင်ခြင်းဖြစ်သည်။ ၎င်းသည် ၎င်းတို့အား model size ပိုကြီးစေနိုင်သည်။ သို့သော်၊ ဤအပို data size ကို embedding matrix တွင် ကျွန်ုပ်တို့ ရရှိသော လျော့ချမှုဖြင့် များစွာသော ကိစ္စများတွင် ဖုံးလွှမ်းသည်၊ ဥပမာ ထိုင်းအတွက် grapheme cluster များအတွက် column 350 အစား code point များအတွက် column 73 အထိသာ ရှိခြင်းဖြင့်။

Code point model များ၏ နောက်ထပ် အားသာချက်တစ်ခု (generalized vector model များနှင့် share လုပ်သည်) မှာ ရှားပါး/မမြင်ဖူးသော grapheme cluster များရှိသော ဝါကျများအတွက် ၎င်းတို့သည် ပိုမိုကောင်းမွန်စွာ perform လုပ်နိုင်ခြင်းဖြစ်သည်။ ထိုကဲ့သို့သော အဖြစ်အပျက်များသည် ကျွန်ုပ်တို့၏ algorithm များကို non-formal text များ၊ ဥပမာ text message များကို segment လုပ်ရန် အသုံးပြုသည့်အခါ မကြာခဏ ဖြစ်ပွားနိုင်သည်။ ဥပမာအားဖြင့်၊ English ဥပမာတစ်ခုကို ပေးရလျှင်၊ တစ်စုံတစ်ယောက်သည် "the meeting is too long" အစား "the meeting is tooooo long" ဟု အသုံးပြုနိုင်သည်၊ ၎င်းတွင် formal word မဟုတ်သော စကားလုံးတစ်လုံး (ထိုင်းတွင် grapheme cluster ဖြစ်နိုင်သည်) ပါဝင်ပြီး၊ ထို့ကြောင့် grapheme cluster ချဉ်းကပ်မှုအတွက် ပြဿနာရှိနိုင်သည်။
