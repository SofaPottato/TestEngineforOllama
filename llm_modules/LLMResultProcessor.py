import pandas as pd
import logging
from pathlib import Path
from typing import List
from .schemas import PipelineError, LabelMapConfig


class LLMResultProcessor:
    def __init__(self, inputCsvPath: Path, outputCsvPath: Path, mergedPath: Path = None,
                 labelMap: LabelMapConfig = None):
        """
        負責將 OutputParser 產出的長表格，清理後轉置為寬表格。

        :param inputCsvPath: OutputParser 產出的結構化 CSV（長表格）
        :param outputCsvPath: 處理後的寬表格 CSV 輸出路徑
        :param mergedPath: 完整版 CSV（含所有自訂欄位）輸出路徑；None 則不產生
        :param labelMap: 標籤映射設定
        """
        self.inputCsvPath = Path(inputCsvPath)
        self.outputCsvPath = Path(outputCsvPath)
        self.mergedPath = Path(mergedPath) if mergedPath else None
        self.labelMap = labelMap

        self.requiredColsList = ['dataID', 'Model', 'promptID', 'predLabel', 'trueLabel']
        self.inputDf = None
        self.pivotDf = None

        logging.info(f"LLMResultProcessor initialized. Input: {self.inputCsvPath}")

    def run(self) -> Path:
        """
        執行完整處理流程：讀取 → 轉換 trueLabel → 建立 Feature_Name → Pivot → 存檔。

        :return: 處理後的寬表格 CSV 路徑
        """
        logging.info(f"Processing data: {self.inputCsvPath}")

        self._loadData()

        logging.info("Converting True Labels...")
        self.inputDf['trueLabel'] = self.inputDf['trueLabel'].apply(self._convertTrueLabel)

        unknownCount = (self.inputDf['trueLabel'] == -1).sum()
        if unknownCount > 0:
            logging.warning(f"Warning: {unknownCount} trueLabel values unrecognized (marked as -1)")

        self.inputDf['Feature_Name'] = self.inputDf['Model'].astype(str) + "_" + self.inputDf['promptID'].astype(str)

        self._pivotData()
        return self._saveData()

    def _loadData(self):
        """讀取並驗證輸入 CSV。"""
        if not self.inputCsvPath.exists():
            raise PipelineError(f"File not found: {self.inputCsvPath}")

        try:
            self.inputDf = pd.read_csv(self.inputCsvPath, encoding='utf-8-sig')
            logging.info(f"Data loaded. Shape: {self.inputDf.shape}")
        except Exception as e:
            raise PipelineError(f"Failed to read CSV: {e}") from e

        missingList = [c for c in self.requiredColsList if c not in self.inputDf.columns]
        if missingList:
            raise PipelineError(f"Missing required columns: {missingList}")

    def _convertTrueLabel(self, x) -> int:
        """
        將原始 trueLabel 標準化為 1/0/-1。

        :param x: 原始標籤值
        :return: 1 (正類)、0 (負類)、-1 (無法識別)
        """
        valStr = str(x).strip().lower()

        if self.labelMap:
            posSet = {v.lower() for v in self.labelMap.positive}
            negSet = {v.lower() for v in self.labelMap.negative}
        else:
            posSet = {'1', 'true', 'yes'}
            negSet = {'0', 'false', 'no', 'none', 'negative'}

        if valStr in posSet:
            return 1
        elif valStr in negSet:
            return 0
        else:
            logging.warning(f"Unrecognized trueLabel: '{x}' -> -1")
            return -1

    def _getFeatureCols(self) -> List[str]:
        """回傳 pivot 後的預測欄（由 Feature_Name 展開而來）。"""
        originalColsSet = set(self.inputDf.columns)
        return [c for c in self.pivotDf.columns if c not in originalColsSet]

    def _pivotData(self):
        """
        將長表格轉置為寬表格。
        動態偵測 index 欄位：除了 Model, promptID, Feature_Name, predLabel, rawOutput 以外的欄位
        都視為 index（dataID, trueLabel, 以及前處理帶入的自訂欄位如 e1, e2）。
        """
        logging.info("Pivoting table (Long to Wide)...")

        nonIndexColsSet = {'Model', 'promptID', 'Feature_Name', 'predLabel', 'rawOutput'}
        indexColsList = [c for c in self.inputDf.columns if c not in nonIndexColsSet]

        try:
            self.pivotDf = self.inputDf.pivot_table(
                index=indexColsList,
                columns='Feature_Name',
                values='predLabel',
                aggfunc='first'
            )
            self.pivotDf = self.pivotDf.reset_index()
            self.pivotDf = self.pivotDf.fillna(-1)
            logging.info(f"Pivot completed. Shape: {self.pivotDf.shape}")
        except Exception as e:
            raise PipelineError(f"Pivot failed: {e}") from e

    def _saveData(self) -> Path:
        """
        儲存結果：
        1. 精簡版（outputCsvPath）：僅含 dataID, trueLabel 與各模型預測欄，供 Evaluate 使用
        2. 完整版（mergedPath）：保留 pivot 的所有欄位（含自訂欄位如 e1/e2），供人工審閱

        :return: 精簡版寬表格路徑
        """
        try:
            predColsList = [c for c in self.pivotDf.columns
                            if c not in ('dataID', 'trueLabel') and c in self._getFeatureCols()]
            leanColsList = [c for c in ('dataID', 'trueLabel') if c in self.pivotDf.columns] + predColsList
            leanDf = self.pivotDf[leanColsList]
            leanDf.to_csv(self.outputCsvPath, index=False, encoding='utf-8-sig')

            if self.mergedPath:
                self.pivotDf.to_csv(self.mergedPath, index=False, encoding='utf-8-sig')
                logging.info(f"Full info saved to: {self.mergedPath}")

            validCount = (self.inputDf['predLabel'] != -1).sum()
            totalCount = len(self.inputDf)

            logging.info(f"Processing complete!")
            logging.info(f"  - Shape: {self.pivotDf.shape}")
            logging.info(f"  - Parse rate: {validCount}/{totalCount} ({validCount/totalCount:.1%})")
            logging.info(f"  - Saved to: {self.outputCsvPath}")

            return self.outputCsvPath

        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"Failed to save results: {e}") from e
