import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report

# 設定 (日本語フォントがないと文字化けするので注意。Windowsなら "MS Gothic" 等指定)
# Linux(WSL)の場合は適宜日本語フォントを入れて指定してください
plt.rcParams['font.family'] = 'sans-serif' 

# 1. モデルとテストデータの読み込み
model_path = "./my_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")

test_df = pd.read_csv("test.csv")
labels_df = pd.read_csv("labels.csv")
id2label = dict(zip(labels_df["id"], labels_df["label"]))

# 2. 推論実行
predictions = []
true_labels = []

model.eval()
print("推論中...")
with torch.no_grad():
    for i, row in test_df.iterrows():
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512).to("cuda")
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(id2label[pred_id])
        true_labels.append(id2label[row["label_id"]])
        
        if i % 100 == 0: print(f"{i}/{len(test_df)}")

# 3. 混同行列の作成と保存
cm = confusion_matrix(true_labels, predictions, labels=list(id2label.values()))

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=list(id2label.values()), yticklabels=list(id2label.values()), cmap="Blues")
plt.ylabel("正解ラベル")
plt.xlabel("予測ラベル")
plt.title("BERTモデルによる分類結果")
plt.savefig("confusion_matrix.png")
print("confusion_matrix.png を保存しました")

# 分類レポート表示
print(classification_report(true_labels, predictions))