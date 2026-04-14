"""
BC5CDR 資料集前處理腳本。
將原始 CSV 轉為標準 Task CSV 格式，供 Pipeline 使用。

標準 Task CSV 欄位：
  - taskID:   唯一識別碼（斷點續傳用）
  - context:  JSON dict，共享欄位（對應 taskTemplate 的佔位符）
  - items:    JSON array，每個 item 帶 id/label 及自訂欄位（對應 itemTemplate 的佔位符）

使用方式：
  # 由 config 讀取所有參數
  python preprocess/bc5cdr.py --config configs/preprocess_bc5cdr.yaml

  # 以 CLI 參數覆寫 config
  python preprocess/bc5cdr.py --config configs/preprocess_bc5cdr.yaml --batch_size 1 --limit 100

  # 不用 config，全部從 CLI 指定
  python preprocess/bc5cdr.py --input data/bcvcdr_raw/BCVCDR_Processed.csv --output data/test/tasks.csv --batch_size 5
"""

import argparse
import json
import logging
import yaml
import pandas as pd
from pathlib import Path


def preprocess(inputPath: str, outputPath: str, batchSize: int, limit: int = None):
    df = pd.read_csv(inputPath, encoding='utf-8-sig', on_bad_lines='warn')

    if limit:
        df = df.head(limit)
        logging.info(f"Testing mode: using first {limit} rows")

    requiredCols = {'ID', 'PMID', 'Title', 'Abstract', 'E1_Name', 'E2_Name', 'Relation_Type'}
    missing = requiredCols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tasks = []

    if batchSize > 1:
        # 批次模式：依 PMID 分組，每組內依 batchSize 切片
        for pmid, group in df.groupby('PMID'):
            context = {
                'title': str(group.iloc[0]['Title']),
                'abstract': str(group.iloc[0]['Abstract'])
            }

            items = [
                {
                    'id': str(row['ID']),
                    'label': str(row['Relation_Type']),
                    'e1': str(row['E1_Name']),
                    'e2': str(row['E2_Name'])
                }
                for _, row in group.iterrows()
            ]

            for i in range(0, len(items), batchSize):
                batchItems = items[i:i + batchSize]
                taskId = f"{pmid}_{i}"
                tasks.append({
                    'taskID': taskId,
                    'context': json.dumps(context, ensure_ascii=False),
                    'items': json.dumps(batchItems, ensure_ascii=False)
                })
    else:
        # 單筆模式：每列獨立一個 task，context 包含所有欄位
        for _, row in df.iterrows():
            context = {
                'title': str(row['Title']),
                'abstract': str(row['Abstract']),
                'e1': str(row['E1_Name']),
                'e2': str(row['E2_Name'])
            }
            items = [{'id': str(row['ID']), 'label': str(row['Relation_Type'])}]
            tasks.append({
                'taskID': str(row['ID']),
                'context': json.dumps(context, ensure_ascii=False),
                'items': json.dumps(items, ensure_ascii=False)
            })

    outPath = Path(outputPath)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    taskDf = pd.DataFrame(tasks)
    taskDf.to_csv(str(outPath), index=False, encoding='utf-8-sig')

    logging.info(f"Preprocessing complete: {len(tasks)} tasks -> {outPath}")
    return taskDf


def loadConfig(configPath: str) -> dict:
    """讀取 YAML config，回傳 dict；檔案不存在則回傳空 dict。"""
    if not configPath:
        return {}
    path = Path(configPath)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {configPath}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

    parser = argparse.ArgumentParser(description="BC5CDR Preprocessor")
    parser.add_argument('--config', type=str, default=None, help='YAML config path')
    parser.add_argument('--input', type=str, default=None, help='Raw CSV path (overrides config)')
    parser.add_argument('--output', type=str, default=None, help='Output Task CSV path (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Items per batch (overrides config)')
    parser.add_argument('--limit', type=int, default=None, help='Limit rows for testing (overrides config)')
    args = parser.parse_args()

    cfg = loadConfig(args.config)

    # CLI 優先；fallback 到 config
    inputPath = args.input if args.input is not None else cfg.get('input')
    outputPath = args.output if args.output is not None else cfg.get('output')
    batchSize = args.batch_size if args.batch_size is not None else cfg.get('batch_size', 1)
    limit = args.limit if args.limit is not None else cfg.get('limit')

    if not inputPath or not outputPath:
        parser.error("必須指定 --input 與 --output（或在 config 中設定）")

    preprocess(inputPath, outputPath, batchSize, limit)
