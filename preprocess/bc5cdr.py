"""
BC5CDR 資料集前處理腳本。
將原始 CSV 轉為標準 Task CSV 格式，供 Pipeline 使用。

標準 Task CSV 欄位：
  - taskID:    唯一識別碼（斷點續傳用），格式為 PMID（依 PMID 分組，每組一筆 task）
  - title:     文章標題（對應 taskTemplate 的 {title} 佔位符）
  - abstract:  文章摘要（對應 taskTemplate 的 {abstract} 佔位符）
  - pairs:     JSON array，包含該 PMID 下所有 entity pair（id/label/e1/e2）

使用方式：
  python preprocess/bc5cdr.py
"""

import json
import logging
import pandas as pd
from pathlib import Path

INPUT_PATH  = "data/bcvcdr_raw/BCVCDR_Processed.csv"
OUTPUT_PATH = "data/test/tasks.csv"


def preprocess():
    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig', on_bad_lines='warn')

    requiredCols = {'ID', 'PMID', 'Title', 'Abstract', 'E1_Name', 'E2_Name', 'Relation_Type'}
    missing = requiredCols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tasks = []
    for pmid, group in df.groupby('PMID', sort=False):
        pairs = [
            {
                'id':    str(row['ID']),
                'label': str(row['Relation_Type']),
                'e1':    str(row['E1_Name']),
                'e2':    str(row['E2_Name'])
            }
            for _, row in group.iterrows()
        ]
        tasks.append({
            'taskID':   str(pmid),
            'title':    str(group.iloc[0]['Title']),
            'abstract': str(group.iloc[0]['Abstract']),
            'pairs':    json.dumps(pairs, ensure_ascii=False)
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
