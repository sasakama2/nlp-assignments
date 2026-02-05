from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# --- パス設定 ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
MODEL_SAVE_DIR = BASE_DIR / "my_bert_model"


# 1. データ読み込み
print(f"データを読み込んでいます: {DATA_DIR}")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
labels_df = pd.read_csv(DATA_DIR / "labels.csv")
num_labels = len(labels_df)

# ラベル列の名前をTrainerに合わせて変更
# label_id -> labels
train_df = train_df.rename(columns={"label_id": "labels"}).drop(columns=["label"])
test_df = test_df.rename(columns={"label_id": "labels"}).drop(columns=["label"])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 2. モデル準備
model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# トークナイズ関数
# テキストをBERT入力用に変換
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 3. 学習設定
# 4060 8GB GPUのため、バッチサイズは16、16ビット精度で学習
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=False,
    bf16=True,
)


# 評価指標
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


# 4. 実行
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("学習開始...")
trainer.train()

model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print(f"学習完了: {MODEL_SAVE_DIR} にモデルを保存しました")
