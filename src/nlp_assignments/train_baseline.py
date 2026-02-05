from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fugashi import Tagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# --- パス設定 ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 1. データ読み込み
print("データ読み込み...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
labels_df = pd.read_csv(DATA_DIR / "labels.csv")

# IDからラベル名への変換辞書
id2label = dict(zip(labels_df["id"], labels_df["label"]))

# 2. 日本語トークナイザの定義 (fugashiを使用)
# BERTの時と同じ辞書(ipadic)
tagger = Tagger("-Owakati")


def tokenize(text):
    # テキストを単語のリストに変換
    return tagger.parse(text).split()


# 3. パイプラインの構築
# TfidfVectorizer: 単語を数値ベクトルに変換
# LinearSVC: 高速でテキスト分類に強いSVM
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, max_features=20000)),
        ("svc", LinearSVC(random_state=42, dual="auto")),
    ]
)

# 4. 学習
print("従来手法(TF-IDF + SVM)で学習中... ")
pipeline.fit(train_df["text"], train_df["label_id"])

# 5. 推論と評価
print("評価中...")
predictions_id = pipeline.predict(test_df["text"])
accuracy = accuracy_score(test_df["label_id"], predictions_id)

print("\n=== 結果 (Baseline: TF-IDF + SVM) ===")
print(f"正解率 (Accuracy): {accuracy:.4f}")

# IDをラベル名に戻す
pred_labels = [id2label[i] for i in predictions_id]
true_labels = [id2label[i] for i in test_df["label_id"]]

# 6. 混同行列の可視化と保存
cm = confusion_matrix(true_labels, pred_labels, labels=list(id2label.values()))

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=list(id2label.values()),
    yticklabels=list(id2label.values()),
    cmap="Reds",
    annot_kws={"size": 14},
)
plt.ylabel("正解ラベル", fontsize=16)
plt.xlabel("予測ラベル", fontsize=16)

save_path = PLOT_DIR / "baseline_confusion_matrix.png"
plt.savefig(save_path)
print(f"画像を保存: {save_path}")

# 詳細レポート
print("\n" + classification_report(true_labels, pred_labels))
