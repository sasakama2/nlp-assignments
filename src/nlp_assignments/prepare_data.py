import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- パス設定 ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "text"
OUTPUT_DIR = BASE_DIR / "data"

# データ読み込み関数
def load_data():
    data = []
    exclude = ["CHANGES.txt", "README.txt", "LICENSE.txt"]
    
    if not DATA_DIR.exists():
        print(f"エラー: データフォルダが見つかりません: {DATA_DIR}")
        return pd.DataFrame(), []

    categories = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"Categories found: {categories}")
    
    for cat in categories:
        cat_dir = DATA_DIR / cat
        files = list(cat_dir.glob("*.txt"))
        
        for f_path in files:
            if f_path.name in exclude: continue
            
            with open(f_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) > 3:
                    text = lines[2].strip() + " " + "".join(lines[3:]).strip()
                    text = text.replace("\n", "").replace("\t", "")
                    data.append({"text": text, "label": cat})
    
    return pd.DataFrame(data), categories

# 実行
df, categories = load_data()

if not df.empty:
    # 出力用ディレクトリを作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label2id = {k: v for v, k in enumerate(categories)}
    df["label_id"] = df["label"].map(label2id)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])

    # dataフォルダ内に保存
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
    pd.DataFrame(list(label2id.items()), columns=["label", "id"]).to_csv(OUTPUT_DIR / "labels.csv", index=False)

    print(f"完了: {OUTPUT_DIR} にCSVファイルを作成しました。")
else:
    print("データの読み込みに失敗しました。")