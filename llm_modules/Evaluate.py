import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from .utils import sanitizeFilename


class PromptCmbEval:
    """
    各 (model, promptID) 組合的分類效能評估器。
    partialInfo.csv → eval_summary.csv + CM 圖 + 熱圖 + samples_to_review.csv。
    """

    _VALID_LABELS = [0, 1]
    _CSV_KWARGS = {'index': False, 'encoding': 'utf-8-sig'}

    def __init__(self, partialInfoCsvPath: Path, outputDirPath: Path = Path("./output")):
        # I/O 延後到 run()，避免 import 期觸發 I/O 副作用
        self.partialInfoCsvPath = Path(partialInfoCsvPath)
        self.outputDirPath = Path(outputDirPath)
        self.plotsDirPath = self.outputDirPath / "plots"

        self.inputDf = None
        self.predColNamesList = []
        self.idColNamesList = []
        self.yTrueLabelSeries = None

        self.metricsResultsList = []
        self.metricsSummaryDf = None
        self.correctnessMatrixDf = None
        self.hardSamplesDf = None
        self.upperBound = 0.0

    def run(self) -> Path:
        """評估入口：讀檔 → 計算指標 → 難題分析 → 繪圖 → 存檔。"""
        self._loadData()
        self._evalAllPredCols()
        self._analyzeUpperBound()
        self._plotConfusionMatrices()
        self._plotHeatmap()
        self._saveResults()
        return self.outputDirPath

    # ── 私有流程方法 ─────────────────────────────────────────────────────────

    def _loadData(self):
        if not self.partialInfoCsvPath.exists():
            raise FileNotFoundError(f"Eval input CSV not found: {self.partialInfoCsvPath}")

        self.inputDf = pd.read_csv(str(self.partialInfoCsvPath))

        self.idColNamesList = ['itemID']
        self.predColNamesList = [col for col in self.inputDf.columns if col not in ('trueLabel', 'itemID')]

        self.yTrueLabelSeries = self.inputDf['trueLabel']
        self.correctnessMatrixDf = pd.DataFrame(index=self.inputDf.index)

        self.plotsDirPath.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"[Eval] 載入完成: shape={self.inputDf.shape}, "
            f"預測欄={len(self.predColNamesList)}, index欄={self.idColNamesList} → {self.outputDirPath}"
        )

    def _evalAllPredCols(self):
        """
        遍歷所有預測欄，計算指標並記錄每個樣本的對錯矩陣。
        指標計算只用有效預測（predLabel ∈ {0,1}）；對錯矩陣用全體樣本（含 -1，-1 一律判錯）。
        """
        for predColName in self.predColNamesList:
            validLabelsTuple = self._getValidPair(predColName)
            if validLabelsTuple is None:
                logging.warning(f"[Eval] 跳過 {predColName}: 無有效預測 (predLabel ∉ {{0,1}})")
                continue
            trueValidSeries, predValidSeries = validLabelsTuple

            evalMetricsDict = self._calcMetrics(trueValidSeries, predValidSeries)
            if evalMetricsDict:
                resultRowDict = {"modelPromptID": predColName}
                resultRowDict.update(evalMetricsDict)
                resultRowDict["validCount"] = len(trueValidSeries)
                self.metricsResultsList.append(resultRowDict)

            # 對錯矩陣含 -1：-1 vs 任何值都為 False，使難題定義不依賴解析成功率
            self.correctnessMatrixDf[predColName] = (self.inputDf[predColName] == self.yTrueLabelSeries).astype(int)

        if self.metricsResultsList:
            self.metricsSummaryDf = pd.DataFrame(self.metricsResultsList).sort_values('f1Score', ascending=False)
        else:
            logging.warning("[Eval] 無有效結果，未產生 eval_summary.csv")

    def _calcMetrics(self, trueLabelSeries, predLabelSeries) -> dict:
        """
        計算單一 runKey 的分類指標（Accuracy / Precision / Recall / F1 / MCC）。
        MCC 對類別不平衡更具參考價值；zero_division=0 讓無正類預測時回傳 0 而非報錯。
        yTrue 為空時回傳 None。
        """
        if len(trueLabelSeries) == 0:
            return None

        metricsDict = {
            "Accuracy":  accuracy_score(trueLabelSeries, predLabelSeries),
            "Precision": precision_score(trueLabelSeries, predLabelSeries, zero_division=0),
            "Recall":    recall_score(trueLabelSeries, predLabelSeries, zero_division=0),
            "f1Score":   f1_score(trueLabelSeries, predLabelSeries, zero_division=0),
            "MCC":       matthews_corrcoef(trueLabelSeries, predLabelSeries)
        }
        return {k: round(v, 2) for k, v in metricsDict.items()}

    def _analyzeUpperBound(self):
        """
        計算難題（所有 runKey 都答錯的樣本）與理論上限。
        Upper Bound = (總樣本 - 難題) / 總樣本，反映「完美解非難題」時的天花板準確率。
        Upper Bound 遠低於目標時，加 prompt 試誤無效，需從資料/模型本身改進。
        """
        if self.correctnessMatrixDf.empty:
            logging.warning("[Eval] correctness matrix 為空，跳過難題分析")
            return

        correctCountsSeries = self.correctnessMatrixDf.sum(axis=1)
        hardSampleIndexList = correctCountsSeries[correctCountsSeries == 0].index

        reviewColNamesList = self.idColNamesList + ['trueLabel']
        availableReviewColNamesList = [c for c in reviewColNamesList if c in self.inputDf.columns]
        self.hardSamplesDf = self.inputDf.loc[hardSampleIndexList, availableReviewColNamesList]

        totalSampleCount = len(self.inputDf)
        solvableSampleCount = totalSampleCount - len(self.hardSamplesDf)
        self.upperBound = solvableSampleCount / totalSampleCount if totalSampleCount > 0 else 0

        logging.info(f"[Eval] 難題分析完成: Upper Bound={self.upperBound:.2%}, 難題 {len(self.hardSamplesDf)} 筆")

    def _plotConfusionMatrices(self):
        """為每個 runKey 繪製混淆矩陣 PNG（排除 -1），存至 plots/ 目錄。"""
        logging.info("[Eval] 繪製混淆矩陣中")

        for predColName in self.predColNamesList:
            validLabelsTuple = self._getValidPair(predColName)
            if validLabelsTuple is None:
                continue
            yTrueValidSeries, yPredValidSeries = validLabelsTuple

            # labels=[0,1] 確保即使某類別無預測，矩陣仍為 2×2
            confusionMatrixArr = confusion_matrix(yTrueValidSeries, yPredValidSeries, labels=self._VALID_LABELS)

            plt.figure(figsize=(6, 5))
            sns.heatmap(confusionMatrixArr, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Pred: 0', 'Pred: 1'],
                        yticklabels=['True: 0', 'True: 1'])
            plt.title(f"Confusion Matrix: {predColName}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()

            savePath = self.plotsDirPath / f"CM_{sanitizeFilename(predColName)}.png"
            plt.savefig(str(savePath), bbox_inches='tight')
            plt.close()

    def _plotHeatmap(self):
        """繪製所有 runKey 對每個樣本的對錯熱圖（綠=對、紅=錯），存至 outputDir。"""
        if self.correctnessMatrixDf.empty:
            logging.warning("[Eval] correctness matrix 為空，跳過熱圖")
            return

        logging.info("[Eval] 繪製對錯熱圖中")
        plt.figure(figsize=(12, 8))
        # .T 轉置：模型放 Y 軸、樣本放 X 軸，符合閱讀直覺
        sns.heatmap(self.correctnessMatrixDf.T, cmap="RdYlGn", cbar=True,
                    cbar_kws={'label': 'Correct (1) / Incorrect (0)'})
        plt.title("Model Correctness Heatmap (Green=Correct)")
        plt.xlabel("Sample Index")
        plt.ylabel("Models")
        plt.tight_layout()
        savePath = self.outputDirPath / "correctness_heatmap.png"
        plt.savefig(str(savePath), bbox_inches='tight')
        plt.close()

    def _saveResults(self):
        """輸出 eval_summary.csv（按 F1 排序）與 samples_to_review.csv（難題清單）。"""
        if self.metricsSummaryDf is not None:
            self.metricsSummaryDf.to_csv(str(self.outputDirPath / "eval_summary.csv"), **self._CSV_KWARGS)

        if self.hardSamplesDf is not None:
            self.hardSamplesDf.to_csv(str(self.outputDirPath / "samples_to_review.csv"), **self._CSV_KWARGS)

        logging.info(f"[Eval] 所有結果已儲存 → {self.outputDirPath}")

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _getValidPair(self, col: str):
        """回傳 (yTrueValidSeries, yPredValidSeries)，若無有效預測（值不在 {0,1}）則 None。"""
        yPredSeries = self.inputDf[col]
        validMaskSeries = yPredSeries.isin(self._VALID_LABELS)
        if validMaskSeries.sum() == 0:
            return None
        return self.yTrueLabelSeries[validMaskSeries], yPredSeries[validMaskSeries]
