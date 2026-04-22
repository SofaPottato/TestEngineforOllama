import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from typing import Iterable, Optional
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 與預測欄無關、無論資料集都會出現的固定 index 欄
_BASE_INDEX_COLS = frozenset({'dataID', 'originalLabel'})

class PromptCmbEval:
    def __init__(self, inputCsvPath: Path, outputBaseDir: Path = Path("./output"),
                 contextColumns: Optional[Iterable[str]] = None):
        """
        初始化評估系統。
        建構子只設定欄位，實際讀檔與產生輸出目錄延後到 run() 進行，
        方便單元測試與避免 import 期就觸發 I/O 副作用。

        :param inputCsvPath: LLMResultProcessor 產出的寬表格 CSV
        :param outputBaseDir: 評估結果（CSV 報表與圖表）的輸出根目錄
        :param contextColumns: 來自 config 的 context/pair 欄名（用於白名單化 index 欄，避免數值型欄位被誤判為預測欄）
        """
        self.inputCsvPath = Path(inputCsvPath)
        self.outputDirPath = Path(outputBaseDir)
        self.plotsDirPath = self.outputDirPath / "plots"
        self.contextColumns = set(contextColumns) if contextColumns else set()

        self.inputDf = None
        self.predColsList = []
        self.indexColsList = []
        self.fixedColsList = []
        self.yTrue = None

        self.resultsList = []
        self.reportDf = None
        self.correctnessMatrixDf = None
        self.hardSamplesDf = None
        self.upperBound = 0.0

    def _loadData(self):
        """讀取輸入 CSV 並以白名單分離 index 欄與預測欄。"""
        if not self.inputCsvPath.exists():
            raise FileNotFoundError(f"Eval input CSV not found: {self.inputCsvPath}")

        self.inputDf = pd.read_csv(str(self.inputCsvPath))

        # 白名單：base + 來自 config 的 contextColumns + pair 展開欄（e1, e2, chemical, disease, ...）
        # 任何不在白名單也不是 trueLabel 的欄都視為預測欄；避免「值域剛好是 0/1」的數值欄被誤判
        indexWhitelist = _BASE_INDEX_COLS | self.contextColumns
        for col in self.inputDf.columns:
            if col == 'trueLabel':
                continue
            if col in indexWhitelist:
                self.indexColsList.append(col)
            else:
                self.predColsList.append(col)

        self.fixedColsList = self.indexColsList + ['trueLabel']
        self.yTrue = self.inputDf['trueLabel']
        self.correctnessMatrixDf = pd.DataFrame(index=self.inputDf.index)

        self.plotsDirPath.mkdir(parents=True, exist_ok=True)

        logging.info(f"LLMEvaluationSystem(inputCsvPath='{self.inputCsvPath}', df_shape={self.inputDf.shape}, pred_cols_count={len(self.predColsList)})")
        logging.info(f"  index cols: {self.indexColsList}")
        logging.info(f"System Initialized. Output directory: {self.outputDirPath}")

    def doCalcPromptCmbMetrics(self, yTrueSubset, yPredSubset):
        """
        計算單一模型/prompt 組合的分類評估指標。

        :param yTrueSubset: 過濾後的真實標籤 Series（已排除 -1 的無效樣本）
        :param yPredSubset: 對應的預測標籤 Series
        :return: 包含 Accuracy/Precision/Recall/F1/MCC 的 dict，各值取到小數點後 2 位；資料為空時回傳 None
        """
        if len(yTrueSubset) == 0:
            return None

        metricsDict = {
            "Accuracy": accuracy_score(yTrueSubset, yPredSubset),
            "Precision": precision_score(yTrueSubset, yPredSubset, zero_division=0),  # 無正類預測時不報錯，回傳 0
            "Recall": recall_score(yTrueSubset, yPredSubset, zero_division=0),
            "f1Score": f1_score(yTrueSubset, yPredSubset, zero_division=0),
            "MCC": matthews_corrcoef(yTrueSubset, yPredSubset)  # MCC 對類別不平衡的資料集更具參考價值
        }
        return {k: round(v, 2) for k, v in metricsDict.items()}

    def doEval(self):
        """
        執行主要評估迴圈：遍歷所有模型/prompt 欄位，計算指標並記錄每個樣本的對錯情況。

        評估結果存入 self.resultsList（後轉為 self.reportDf），
        各樣本對錯記錄存入 self.correctnessMatrixDf，供 doAnalyzeUpperBound 使用。
        """
        for col in self.predColsList:
            yPred = self.inputDf[col]

            # 排除 -1（解析失敗或任務跳過），只對有效預測值計算指標
            validMask = yPred.isin([0, 1])
            if validMask.sum() == 0:
                logging.warning(f"⚠️ Warning: Model '{col}' has no valid predictions (0 or 1). Skipping.")
                continue

            yTrueValid = self.yTrue[validMask]
            yPredValid = yPred[validMask]

            metricsDict = self.doCalcPromptCmbMetrics(yTrueValid, yPredValid)
            if metricsDict:
                resDict = {"modelPromptID": col}
                resDict.update(metricsDict)
                resDict["validCount"] = len(yTrueValid)  # 紀錄有效預測數，便於判斷結果可信度
                self.resultsList.append(resDict)

            # 以全體樣本（含 -1）計算對錯，-1 vs 任何值都為 False (0)，不影響難題定義
            isCorrectSeries = (yPred == self.yTrue).astype(int)
            self.correctnessMatrixDf[col] = isCorrectSeries

        if self.resultsList:
            self.reportDf = pd.DataFrame(self.resultsList)
            self.reportDf = self.reportDf.sort_values('f1Score', ascending=False)  # 按 F1 降序，方便找最佳組合
        else:
            logging.error("❌ No valid results generated.")

    def doAnalyzeUpperBound(self):
        """
        計算難題（所有模型均答錯的樣本）與理論上限（Upper Bound）。

        Upper Bound = (總樣本數 - 難題數) / 總樣本數
        代表即使完美解決所有非難題，理論上能達到的最高準確率。

        結果存入 self.hardSamplesDf 與 self.upperBound。
        """
        if self.correctnessMatrixDf.empty:
            logging.warning("Correctness matrix is empty. Skipping difficulty analysis.")
            return

        # 對每個樣本橫向加總對錯值，加總為 0 代表所有模型都答錯
        correctCountsSeries = self.correctnessMatrixDf.sum(axis=1)
        hardIndicesIdx = correctCountsSeries[correctCountsSeries == 0].index

        reviewColsList = self.indexColsList + ['trueLabel']
        availableReviewColsList = [c for c in reviewColsList if c in self.inputDf.columns]
        self.hardSamplesDf = self.inputDf.loc[hardIndicesIdx, availableReviewColsList]

        totalSamples = len(self.inputDf)
        solvableSamples = totalSamples - len(self.hardSamplesDf)
        self.upperBound = solvableSamples / totalSamples if totalSamples > 0 else 0

        logging.info(f"Difficulty Analysis Complete. Upper Bound: {self.upperBound:.2%} (Found {len(self.hardSamplesDf)} hard samples)")

    def doPlotConfusionMatrices(self):
        """
        為每個模型/prompt 組合繪製混淆矩陣並儲存為 PNG。
        僅使用有效預測值（0 或 1），排除解析失敗的 -1。

        圖表儲存至 self.plotsDirPath，檔名格式：CM_{safeFeatureName}.png
        """
        logging.info("============Generating Confusion Matrices============")

        for col in self.predColsList:
            yPred = self.inputDf[col]
            validMask = yPred.isin([0, 1])
            if validMask.sum() == 0: continue

            yTrueValid = self.yTrue[validMask]
            yPredValid = yPred[validMask]

            # 明確指定 labels=[0, 1] 確保即使某類別無預測，矩陣仍為 2×2
            confusionMatrixArr = confusion_matrix(yTrueValid, yPredValid, labels=[0, 1])

            plt.figure(figsize=(6, 5))
            sns.heatmap(confusionMatrixArr, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Pred: 0', 'Pred: 1'],
                        yticklabels=['True: 0', 'True: 1'])
            plt.title(f"Confusion Matrix: {col}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()

            # 將欄名中的特殊字元替換，確保檔名在各 OS 均有效
            safeFileNameStr = str(col).replace(":", "_").replace("+", "_").replace(" ", "_").replace("/", "_")
            savePath = self.plotsDirPath / f"CM_{safeFileNameStr}.png"

            plt.savefig(str(savePath), bbox_inches='tight')
            plt.close()

    def doPlotHeatmap(self):
        """
        繪製所有模型對每個樣本的對錯分佈熱圖（Correctness Heatmap）並儲存。

        X 軸為樣本索引，Y 軸為模型/prompt 組合；
        綠色代表答對，紅色代表答錯，整體視覺化各組合在哪些樣本上表現不一致。

        圖表儲存至 self.outputDirPath / correctness_heatmap.png
        """
        if self.correctnessMatrixDf.empty:
            logging.warning("Correctness matrix is empty. Skipping heatmap plotting.")
            return

        logging.info("============Generating Correctness Heatmap============")
        plt.figure(figsize=(12, 8))
        # .T 轉置：將模型放在 Y 軸（行）、樣本放在 X 軸（列），符合直覺的閱讀方向
        sns.heatmap(self.correctnessMatrixDf.T, cmap="RdYlGn", cbar=True, cbar_kws={'label': 'Correct (1) / Incorrect (0)'})
        plt.title("Model Correctness Heatmap (Green=Correct)")
        plt.xlabel("Sample Index")
        plt.ylabel("Models")
        plt.tight_layout()
        savePath = self.outputDirPath / "correctness_heatmap.png"

        plt.savefig(str(savePath), bbox_inches='tight')
        plt.close()

    def doSaveResults(self):
        """
        輸出所有 CSV 報表至 self.outputDirPath：
        - eval_summary.csv：各模型/prompt 組合的評估指標（按 F1 排序）
        - samples_to_review.csv：所有模型均答錯的難題清單，供人工審閱
        """
        if self.reportDf is not None:
            self.reportDf.to_csv(str(self.outputDirPath / "eval_summary.csv"), index=False, encoding='utf-8-sig')

        if self.hardSamplesDf is not None:
            self.hardSamplesDf.to_csv(str(self.outputDirPath / "samples_to_review.csv"), index=False, encoding='utf-8-sig')

        logging.info(f"✅ All results saved to: {self.outputDirPath}")

    def run(self) -> Path:
        """
        評估階段的統一入口：依序載入資料、執行指標計算、理論上限分析、圖表輸出與結果存檔。
        外部只需呼叫這個方法，不需關心內部步驟的順序。

        :return: 評估結果的輸出目錄（供下游引用）
        """
        self._loadData()
        self.doEval()
        self.doAnalyzeUpperBound()
        self.doPlotConfusionMatrices()
        self.doPlotHeatmap()
        self.doSaveResults()
        return self.outputDirPath
