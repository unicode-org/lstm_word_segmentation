Repository ကို Google Cloud Platform နှင့် ချိတ်ဆက်ရန်၊ အောက်ပါအဆင့်များကို လိုက်နာပါ-

1.  Google Cloud API & Services တွင်၊ အောက်ပါတို့ကို enable လုပ်ပါ
    *   Secret Manager API
    *   Artifact Registry API
    *   Vertex AI API
    *   Cloud Pub/Sub API
    *   Cloud Build API

2.  Google Cloud Storage Bucket တစ်ခုကို ဖန်တီးပြီး၊ dataset ကို gs://bucket_name/Data/ သို့ အောက်ပါ directory structure ဖြင့် upload လုပ်ပါ-
    Data/
    ├── Best/
    │ ├── article/
    │ ├── encyclopedia/
    │ ├── news/
    │ └── novel/
    ├── my_test_segmented.txt
    ├── my_train.txt
    └── my_valid.txt

3.  Artifact Registry တွင်၊ storage bucket နှင့် တူညီသော region တွင် repository ကို ဖန်တီးပါ။

4.  Cloud Build တွင်၊ Artifact Registry နှင့် တူညီသော region တွင် trigger တစ်ခုကို ဖန်တီးပါ။
    *   သင့်တော်သော event တစ်ခုကို ရွေးချယ်ပါ (ဥပမာ- Push to a branch)
    *   2nd gen repository generation ကို ရွေးချယ်ပါ
    *   GitHub repository ကို ချိတ်ဆက်ပါ
    *   Dockerfile (Configurations အတွက်) နှင့် Repository (Location အတွက်) ကို ရွေးချယ်ပါ
    *   Dockerfile name: ```Dockerfile```, image name: ```us-central1-docker.pkg.dev/project-name/registry-name/image:latest```
    *   "Require approval before build executes" ကို Enable လုပ်ပါ
    *   Manual image build အတွက်၊ ဖန်တီးထားသော trigger တွင် Enable/ Run ကို နှိပ်ပါ

5.  Image ကို ဖန်တီးပြီး Artifact Registry တွင် သိမ်းဆည်းပြီးနောက်၊ Vertex AI ရှိ Training tab အောက်တွင် "Train new model" ကို ရွေးချယ်ပါ။
    *   Training method: default (Custom training) နှင့် continue
    *   Model details: name ကို ဖြည့်ပြီး continue
    *   Training container: custom container ကို ရွေးချယ်ပြီး နောက်ဆုံး build လုပ်ထားသော image ကို browse လုပ်ပါ၊ storage bucket သို့ link လုပ်ပြီး arguments အောက်တွင်၊ အောက်ပါတို့ကို ပြင်ဆင်ပြီး paste လုပ်ပါ
        ```
        --path=gs://bucket_name/Data/
        --language=Thai
        --input-type=BEST
        --model-type=cnn
        --epochs=200
        --filters=32
        --name=Thai_codepoints_32
        --edim=40
        --embedding=codepoints
        ```
    *   Hyperparameters: unselect လုပ်ပြီး continue
    *   Compute and pricing: ရှိပြီးသား resource များကို ရွေးချယ်ပါ သို့မဟုတ် worker pool အသစ်သို့ deploy လုပ်ပါ
    *   Prediction container: no prediction container နှင့် start training
