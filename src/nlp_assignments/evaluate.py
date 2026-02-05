import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib  # 【NEW】インポートするだけで日本語対応完了
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# --- パス設定 ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# 学習済みモデルのパス (train_bert.pyの設定と合わせる)
MODEL_DIR = BASE_DIR / "my_bert_model"

# 【NEW】画像の出力先フォルダ
PLOT_DIR = BASE_DIR / "plots"
# ----------------

# プロット出力用フォルダを作成
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 読み込み
print(f"モデルを読み込んでいます: {MODEL_DIR}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to("cuda")
except OSError:
    print(f"エラー: モデルが見つかりません ({MODEL_DIR})。先に train_bert.py を実行してください。")
    exit()

test_df = pd.read_csv(DATA_DIR / "test.csv")
labels_df = pd.read_csv(DATA_DIR / "labels.csv")
id2label = dict(zip(labels_df["id"], labels_df["label"]))

# 2. 推論
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
        true_labels.append(id2label[row["label_id"]]) # train_bert.pyでrenameする前のCSVを使うため label_id のまま
        
        if i % 100 == 0: print(f"{i}/{len(test_df)}")

# 3. 可視化
cm = confusion_matrix(true_labels, predictions, labels=list(id2label.values()))

plt.figure(figsize=(12, 10))
# japanize_matplotlib が入っていれば、日本語は自動で表示されます
sns.heatmap(cm, annot=True, fmt="d", xticklabels=list(id2label.values()), yticklabels=list(id2label.values()), cmap="Blues")
plt.ylabel("正解ラベル")
plt.xlabel("予測ラベル")
plt.title("BERTモデルによる分類結果")

# 画像保存
save_path = PLOT_DIR / "confusion_matrix.png"
plt.savefig(save_path)
print(f"画像を保存しました: {save_path}")

# 分類レポート表示
print(classification_report(true_labels, predictions))