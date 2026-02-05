import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# 1. データ読み込み
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
labels_df = pd.read_csv("labels.csv")
num_labels = len(labels_df)

# dataset形式に変換
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 2. トークナイザとモデルの準備 (東北大BERT v3)
model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # max_length=512で切り詰め、paddingする
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# トークン化実行
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# モデル初期化 (GPUへはTrainerが勝手に送ってくれます)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 3. 学習設定
training_args = TrainingArguments(
    output_dir="./results",          # 出力フォルダ
    eval_strategy="epoch",     # エポックごとに評価
    learning_rate=2e-5,              # 学習率
    per_device_train_batch_size=16,  # 4060 (8GB) なら16でいけるはず。ダメなら8に。
    per_device_eval_batch_size=16,
    num_train_epochs=3,              # 3エポックで十分収束します
    weight_decay=0.01,
    save_total_limit=1,              # 容量節約のため最新モデルだけ残す
    fp16=True,                       # GPUの高速化・省メモリ機能 (RTXシリーズなら必須)
)

# 精度計算用関数
from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 4. Trainerの作成と実行
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("学習開始...")
trainer.train()

# モデルの保存
model.save_pretrained("./my_bert_model")
tokenizer.save_pretrained("./my_bert_model")
print("学習完了＆保存しました")
