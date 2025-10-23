# Grapheme Cluster Frequency နှင့် Word Boundary ဆက်စပ်မှု သုံးသပ်ချက်

ဤ Document တွင် `Public_Data` folder ထဲရှိ `.npy` ဖိုင်များ၏ အချက်အလက်များနှင့် Word Segmentation Model ၏ Word Boundary (စကားလုံး နယ်နိမိတ်) သတ်မှတ်ခြင်းတို့ မည်သို့ ဆက်စပ်နေသည်ကို အသေးစိတ် ရှင်းလင်းတင်ပြထားပါသည်။

## 1. `Public_Data` Folder ရှိ `.npy` ဖိုင်များ

`Public_Data` folder ထဲတွင် `Burmese_graph_clust_ratio.npy` ကဲ့သို့သော `.npy` ဖိုင်များ ပါဝင်ပါသည်။ ဤဖိုင်များသည် NumPy library ကို အသုံးပြု၍ သိမ်းဆည်းထားသော binary ဖိုင်များ ဖြစ်ကြသည်။

အတွင်းတွင် **Dictionary** ပုံစံဖြင့် data များကို သိမ်းဆည်းထားပြီး၊ Key မှာ **Grapheme Cluster** (စာလုံး) ဖြစ်ပြီး၊ Value မှာ ထို Grapheme Cluster ၏ **ဖြစ်ပွားနှုန်း အချိုး (Frequency Ratio)** ဖြစ်ပါသည်။

**ဥပမာ (`Burmese_graph_clust_ratio.npy` မှ):**
*   Grapheme Cluster: ' ' (space), Ratio: 0.1355
*   Grapheme Cluster: 'း', Ratio: 0.0834
*   Grapheme Cluster: 'ာ', Ratio: 0.0704

ဤ frequency ratio များကို corpus (စာသား data အစုအဝေး) ကြီးတစ်ခုမှ ကြိုတင်တွက်ချက်ထားခြင်း ဖြစ်ပြီး၊ မြန်မာစာတွင် မည်သည့် စာလုံးများက မည်မျှ အသုံးများကြောင်းကို ဖော်ပြပါသည်။

## 2. Grapheme Cluster ဆိုသည်မှာ အဘယ်နည်း။

Grapheme Cluster ဆိုသည်မှာ အသုံးပြုသူတစ်ဦးက စာလုံးတစ်လုံး (single character) အဖြစ် မြင်သာသော text ၏ အသေးဆုံး အစိတ်အပိုင်း ဖြစ်သည်။ ဥပမာ - မြန်မာစာတွင် 'က' သည် grapheme cluster တစ်ခု ဖြစ်သလို၊ 'ကျ' (က + ျ) သည်လည်း grapheme cluster တစ်ခု ဖြစ်ပါသည်။ Model သည် စာကြောင်းတစ်ကြောင်းကို ဤ grapheme cluster များအဖြစ် အရင်ဆုံး ပိုင်းခြားပါသည်။

## 3. Frequency Data များကို Model က မည်သို့ အသုံးပြုသနည်း။

Word Segmentation Model (အထူးသဖြင့် `grapheme_clusters_tf` embedding type ကို အသုံးပြုသော model) သည် training မစတင်မီ ဤ frequency data များကို အသုံးပြု၍ **Embedding Dictionary** တစ်ခု တည်ဆောက်ပါသည်။

`word_segmenter.py` script ထဲရှိ code အရ၊ model သည်:
1.  `.npy` ဖိုင်မှ grapheme cluster နှင့် သူတို့၏ frequency ratio များကို load လုပ်ပါသည်။
2.  Frequency အများဆုံး (အဖြစ်အများဆုံး) grapheme cluster များကို `input_clusters_num` (ဥပမာ - 350) အရေအတွက်အထိ ရွေးချယ်ပါသည်။
3.  ထို ရွေးချယ်ထားသော cluster များကိုသာ model ၏ embedding dictionary ထဲသို့ ထည့်သွင်းပြီး ID တစ်ခုစီ သတ်မှတ်ပေးပါသည်။
4.  ဤ dictionary ထဲတွင် မပါဝင်သော (frequency အလွန်နည်းသော) grapheme cluster များအားလုံးကို **"Unknown Cluster"** အဖြစ် ID တစ်ခုတည်းဖြင့်သာ သတ်မှတ်ပါသည်။

## 4. Frequency နှင့် Word Boundary (BIES Tags) ဆက်စပ်မှု

Grapheme Cluster တစ်ခု၏ frequency သည် ၎င်း၏ word boundary (BIES tag) ဖြစ်နိုင်ခြေကို တိုက်ရိုက် သတ်မှတ်ပေးခြင်း မဟုတ်သော်လည်း၊ model ၏ "သင်ယူနိုင်စွမ်း" အပေါ် သွယ်ဝိုက်သောအားဖြင့် အလွန်အရေးပါသော သက်ရောက်မှုရှိပါသည်။

*   **Frequency မြင့်သော Grapheme Cluster များ (Known Clusters):**
    *   ဤ cluster များ (ဥပမာ - 'ာ', 'း', 'င်', 'အ') သည် training data ထဲတွင် အကြိမ်များစွာ ပါဝင်သောကြောင့် model ၏ embedding dictionary ထဲတွင် ကိုယ်ပိုင် ID တစ်ခုစီ ရရှိကြသည်။
    *   Training ပြုလုပ်သည့်အခါ၊ model သည် ဤ cluster များ၏ ပုံစံအမျိုးမျိုးကို (စကားလုံး၏ အစ၊ အလယ်၊ အဆုံး) အကြိမ်များစွာ တွေ့မြင်ရသောကြောင့်၊ ၎င်းတို့၏ BIES tag ဖြစ်နိုင်ခြေကို ကောင်းမွန်စွာ သင်ယူနိုင်ပါသည်။
    *   **ဥပမာ:** 'င်' ဆိုသော cluster သည် အများအားဖြင့် စကားလုံး၏ အဆုံး (**E**) သို့မဟုတ် အတွင်း (**I**) တွင်သာ လာလေ့ရှိပြီး၊ အစ (**B**) တွင် လာလေ့မရှိသည်ကို model က ကောင်းစွာ သင်ယူနိုင်ပါလိမ့်မည်။ ထို့အတူ 'အ' သည် အများအားဖြင့် စကားလုံး၏ အစ (**B**) သို့မဟုတ် တစ်လုံးတည်း (**S**) အဖြစ် လာတတ်သည်ကို သင်ယူနိုင်ပါမည်။

*   **Frequency နိမ့်သော Grapheme Cluster များ (Unknown Clusters):**
    *   ဤ cluster များ (ဥပမာ - ရှားပါးသော စာလုံးပေါင်းများ၊ typo များ) သည် embedding dictionary ထဲတွင် မပါဝင်ပါ။
    *   Model သည် ဤ cluster အားလုံးကို "Unknown Cluster" ID တစ်ခုတည်းဖြင့်သာ မြင်သောကြောင့်၊ cluster တစ်ခုချင်းစီ၏ တိကျသော BIES pattern ကို သင်ယူနိုင်ခြင်း မရှိပါ။
    *   ထို့ကြောင့်၊ model သည် ဤ "Unknown Cluster" ၏ word boundary ကို ဆုံးဖြတ်ရန် ၎င်း၏ **ရှေ့နှင့်နောက်ရှိ "Known Cluster" များ၏ context** ကိုသာ အဓိက အားကိုးရပါသည်။ ၎င်းကြောင့် တိကျမှန်ကန်မှု အနည်းငယ် လျော့ကျသွားနိုင်ပါသည်။

### ကောက်ချက်

လေ့လာနေသူများအတွက် အချုပ်အားဖြင့်ဆိုသော်၊ grapheme cluster တစ်ခု၏ **ဖြစ်ပွားနှုန်း (frequency) မြင့်မားခြင်း** သည် segmentation model အား ထို cluster နှင့် ပတ်သက်သော **စကားလုံး နယ်နိမိတ် ပုံစံများ (word boundary patterns)** ကို ပိုမို ကောင်းမွန်စွာ သင်ယူနိုင်စေရန် အဓိက ကူညီပေးပါသည်။ ဤ `.npy` ဖိုင်များထဲရှိ frequency data များသည် model အတွက် မည်သည့် cluster များကို အလေးထား သင်ယူသင့်သည်ကို လမ်းညွှန်ပေးသည့် အရေးကြီးသော အချက်အလက်များပင် ဖြစ်ပါသည်။
