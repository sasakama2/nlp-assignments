from pathlib import Path

import pandas as pd
import torch
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- パス設定 ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "my_bert_model"

print("データを読み込んでいます...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
labels_df = pd.read_csv(DATA_DIR / "labels.csv")
id2label = dict(zip(labels_df["id"], labels_df["label"]))

print("SVMを学習中...")
tagger = Tagger("-Owakati")


def tokenize(text):
    return tagger.parse(text).split()


svm_pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_features=20000)),
        ("svc", LinearSVC(random_state=42, dual="auto")),
    ]
)
svm_pipeline.fit(train_df["text"], train_df["label_id"])
svm_preds = svm_pipeline.predict(test_df["text"])

print("BERTで推論中...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to("cuda")
model.eval()

bert_preds = []
with torch.no_grad():
    for i, row in test_df.iterrows():
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512).to("cuda")
        outputs = model(**inputs)
        bert_preds.append(torch.argmax(outputs.logits, dim=1).item())
        if i % 500 == 0:
            print(f"BERT Progress: {i}/{len(test_df)}")


print("\n=== 分析結果: BERTだけが正解した例 ===")
count = 0
for i, (true, svm, bert, text) in enumerate(zip(test_df["label_id"], svm_preds, bert_preds, test_df["text"])):
    if bert == true and svm != true:
        count += 1
        print("-" * 80)
        print(f"【ID: {i}】")
        print(f"正解ラベル: {id2label[true]}")
        print(f"SVMの予測 : {id2label[svm]} (間違い)")
        print(f"BERTの予測: {id2label[bert]} (正解)")
        print(f"記事冒頭  : {text[:100]}...")  # 記事の中身を見る
        if count >= 3:
            break  # 3件見つかったら終了

print("\n" + "=" * 80 + "\n")

print("=== 分析結果: SVMだけが正解した例  ===")
count = 0
for i, (true, svm, bert, text) in enumerate(zip(test_df["label_id"], svm_preds, bert_preds, test_df["text"])):
    if svm == true and bert != true:
        count += 1
        print("-" * 80)
        print(f"【ID: {i}】")
        print(f"正解ラベル: {id2label[true]}")
        print(f"SVMの予測 : {id2label[svm]} (正解)")
        print(f"BERTの予測: {id2label[bert]} (間違い)")
        print(f"記事冒頭  : {text[:100]}...")
        if count >= 3:
            break
