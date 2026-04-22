"""
LLL 資料集前處理腳本。
將 LLL-test.csv 轉為標準 Task CSV 格式，供 Pipeline 使用。

LLL 原始欄位：
  - docid, isValid, passage, passageid

標準 Task CSV 欄位（輸出）：
  - taskID:   唯一識別碼，使用 row index（每句一筆 task）
  - passage:  單句文本（對應 taskTemplate 的 {passage} 佔位符）
  - pairs:    JSON array，只帶 label（實體已固定為 PROTEIN1/PROTEIN2 嵌入句中）

使用方式：
  python preprocess/lll.py
"""

import json
import logging
import pandas as pd
from pathlib import Path

INPUT_PATH  = "data\PPI\LLL-train.csv"
OUTPUT_PATH = "data\PPI\LLL\LLL-tasks.csv"


def preprocess():
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')

    required = {'isValid', 'passage', 'passageid'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tasks = []
    for i, row in df.iterrows():
        label = "true" if str(row['isValid']).strip().upper() == "TRUE" else "false"
        pair = [{"label": label}]
        tasks.append({
            "taskID":  str(row['passageid']) + f"_{i}",
            "passage": str(row['passage']),
            "pairs":   json.dumps(pair, ensure_ascii=False)
        })

    outPath = Path(OUTPUT_PATH)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    taskDf = pd.DataFrame(tasks)
    taskDf.to_csv(str(outPath), index=False, encoding='utf-8-sig')

    logging.info(f"Preprocessing complete: {len(tasks)} tasks -> {outPath}")
    return taskDf


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")
    preprocess()
