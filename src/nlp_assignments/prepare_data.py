import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# 修正箇所: スクリプトのある場所(src/npl-assignments)から見て
# textフォルダは2階層上(root)のtextフォルダにある
DATA_DIR = "../../text"
OUTPUT_FILE = "livedoor_news.csv"

def load_data():
    data = []
    # 除外するファイル
    exclude = ["CHANGES.txt", "README.txt", "LICENSE.txt"]
    
    # パスが存在するか確認
    if not os.path.exists(DATA_DIR):
        print(f"エラー: データフォルダが見つかりません: {os.path.abspath(DATA_DIR)}")
        print("ディレクトリ構造を確認してください。")
        return pd.DataFrame(), []

    categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"Categories found: {categories}")
    
    for cat in categories:
        files = glob.glob(os.path.join(DATA_DIR, cat, "*.txt"))
        for f_path in files:
            if os.path.basename(f_path) in exclude: continue
            
            with open(f_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # 3行目がタイトル、4行目以降が本文
                if len(lines) > 3:
                    text = lines[2].strip() + " " + "".join(lines[3:]).strip()
                    # 空白や改行を簡易除去
                    text = text.replace("\n", "").replace("\t", "")
                    data.append({"text": text, "label": cat})
    
    return pd.DataFrame(data), categories

# 実行
df, categories = load_data()

if not df.empty:
    # ラベルをIDに変換
    label2id = {k: v for v, k in enumerate(categories)}
    df["label_id"] = df["label"].map(label2id)

    # 訓練データとテストデータに分割 (8:2)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])

    # CSVは実行ディレクトリ(src/npl-assignments/)に保存されます
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    pd.DataFrame(list(label2id.items()), columns=["label", "id"]).to_csv("labels.csv", index=False)

    print("データ準備完了！ train.csv, test.csv が作成されました。")
else:
    print("データの読み込みに失敗しました。")