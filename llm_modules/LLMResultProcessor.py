import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict
from .schemas import PipelineError, LabelMapConfig


class LLMResultProcessor:
    """
    將 OutputParser 的長表格清理後轉置為寬表格，供下游 Evaluate 與人工檢視使用。
    result.csv (long) → partialInfo.csv (wide) + fullInfo.csv（含 rawOutput / sysPrompt）。
    """

    _RAW_SUFFIX = '__raw'
    _SYS_PROMPT_SUFFIX = '__sysPrompt'
    _CSV_KWARGS = {'index': False, 'encoding': 'utf-8-sig'}
    _REQUIRED_COLS = ('itemID', 'model', 'promptID', 'predLabel', 'trueLabel')
    _NON_INDEX_COLS = {'model', 'promptID', 'runKey', 'predLabel', 'rawOutput'}

    _DEFAULT_TRUE_POS = {'1', 'true', 'yes'}
    _DEFAULT_TRUE_NEG = {'0', 'false', 'no', 'none', 'negative'}

    def __init__(self, parsedOutputCsvPath: Path, partialInfoCsvPath: Path, fullInfoCsvPath: Path = None,
                 labelMapConfig: LabelMapConfig = None, promptCmbList: List[Dict] = None):
        self.parsedOutputCsvPath = Path(parsedOutputCsvPath)
        self.partialInfoCsvPath = Path(partialInfoCsvPath)
        self.fullInfoCsvPath = Path(fullInfoCsvPath) if fullInfoCsvPath else None
        self.promptCmbList = promptCmbList or []

        # 預先計算 trueLabel 的正負集合，避免每筆轉換都重建
        if labelMapConfig:
            self._truePosSet = {v.lower() for v in labelMapConfig.positive}
            self._trueNegSet = {v.lower() for v in labelMapConfig.negative}
        else:
            self._truePosSet = self._DEFAULT_TRUE_POS
            self._trueNegSet = self._DEFAULT_TRUE_NEG

        self.inputDf = None
        self.pivotDf = None
        self.fullPivotDf = None

    def run(self) -> Path:
        """主流程協調者：讀取 → 標準化 → Pivot → 存檔，回傳精簡版寬表格路徑。"""
        logging.info(f"[Processor] 啟動: {self.parsedOutputCsvPath}")
        self._loadData()
        self._prepareDf()
        self._pivotData()
        return self._saveData()

    # ── 私有流程方法 ─────────────────────────────────────────────────────────

    def _loadData(self):
        """讀取並驗證輸入 CSV，缺必要欄位即拋 PipelineError。"""
        if not self.parsedOutputCsvPath.exists():
            raise PipelineError(f"File not found: {self.parsedOutputCsvPath}")
        try:
            self.inputDf = pd.read_csv(self.parsedOutputCsvPath, encoding='utf-8-sig')
        except Exception as e:
            raise PipelineError(f"Failed to read CSV: {e}") from e

        missingList = [c for c in self._REQUIRED_COLS if c not in self.inputDf.columns]
        if missingList:
            raise PipelineError(f"Missing required columns: {missingList}")

    def _prepareDf(self):
        """備份 originalLabel、標準化 trueLabel、組 runKey。"""
        self.inputDf['originalLabel'] = self.inputDf['trueLabel']
        self.inputDf['trueLabel'] = self.inputDf['trueLabel'].apply(self._convertTrueLabel)

        unknownCount = (self.inputDf['trueLabel'] == -1).sum()
        if unknownCount > 0:
            logging.warning(f"[Processor] 共 {unknownCount} 筆 trueLabel 未識別")

        self.inputDf['runKey'] = (self.inputDf['model'].astype(str) + "_" +
                                  self.inputDf['promptID'].astype(str))

    def _convertTrueLabel(self, x) -> int:
        """將 trueLabel 標準化為 1 / 0 / -1。正負集合在 __init__ 已預先計算。"""
        valStr = str(x).strip().lower()
        if valStr in self._truePosSet:
            return 1
        if valStr in self._trueNegSet:
            return 0
        logging.warning(f"[Processor] 未識別的 trueLabel: '{x}' → -1")
        return -1

    def _pivotData(self):
        """
        long → wide pivot：以樣本為列、runKey 為欄、predLabel 為值。
        index 欄動態偵測（排除 _NON_INDEX_COLS），讓上游新增資料欄不需要改本模組。
        同時產出 pivotDf（predLabel）與 fullPivotDf（predLabel + __raw 後綴欄）。
        """
        indexColsList = [c for c in self.inputDf.columns if c not in self._NON_INDEX_COLS]

        try:
            predPivotDf = self.inputDf.pivot_table(
                index=indexColsList,
                columns='runKey',
                values='predLabel',
                aggfunc='first'
            ).fillna(-1)
            self.pivotDf = predPivotDf.reset_index()

            rawPivotDf = self.inputDf.pivot_table(
                index=indexColsList,
                columns='runKey',
                values='rawOutput',
                aggfunc='first'
            ).fillna('')
            rawPivotDf.columns = [f"{c}{self._RAW_SUFFIX}" for c in rawPivotDf.columns]

            self.fullPivotDf = pd.concat([predPivotDf, rawPivotDf], axis=1).reset_index()
        except Exception as e:
            raise PipelineError(f"Pivot failed: {e}") from e

    def _saveData(self) -> Path:
        """寫精簡版（必出）與完整版（若提供路徑），回傳精簡版路徑。"""
        try:
            self._buildLeanDf().to_csv(self.partialInfoCsvPath, **self._CSV_KWARGS)

            if self.fullInfoCsvPath and self.fullPivotDf is not None:
                sysPromptCols = self._appendSysPromptCols()
                orderedDf = self._orderFullDfColumns(sysPromptCols)
                orderedDf.to_csv(self.fullInfoCsvPath, **self._CSV_KWARGS)

            self._logRunStats()
            return self.partialInfoCsvPath
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"Failed to save results: {e}") from e

    # ── _saveData 的子步驟 ───────────────────────────────────────────────────

    def _buildLeanDf(self) -> pd.DataFrame:
        """精簡版：itemID + trueLabel + 各 runKey 的 predLabel 欄。"""
        predCols = self._getFeatureCols()
        leanCols = [c for c in ('itemID', 'trueLabel') if c in self.pivotDf.columns] + predCols
        return self.pivotDf[leanCols]

    def _appendSysPromptCols(self) -> List[str]:
        """為 fullPivotDf 補 {runKey}__sysPrompt 欄，回傳新增欄位名清單。"""
        sysPromptCols: List[str] = []
        if not self.promptCmbList:
            return sysPromptCols

        promptIDToText = {p['promptID']: p['promptText'] for p in self.promptCmbList}
        runKeyIter = (self.inputDf[['model', 'promptID', 'runKey']]
                      .drop_duplicates()
                      .itertuples(index=False))
        for row in runKeyIter:
            colName = f"{row.runKey}{self._SYS_PROMPT_SUFFIX}"
            self.fullPivotDf[colName] = promptIDToText.get(row.promptID, '')
            if colName not in sysPromptCols:
                sysPromptCols.append(colName)
        return sysPromptCols

    def _orderFullDfColumns(self, sysPromptCols: List[str]) -> pd.DataFrame:
        """
        完整版欄位重排：itemID → labels → __raw → pred → __sysPrompt → 其他 index 欄。
        固定的閱讀順序方便人工開檔對照。
        """
        originalColsSet = set(self.inputDf.columns)
        predCols = [c for c in self.fullPivotDf.columns
                    if not c.endswith(self._RAW_SUFFIX)
                    and not c.endswith(self._SYS_PROMPT_SUFFIX)
                    and c not in originalColsSet]
        rawCols = [c for c in self.fullPivotDf.columns if c.endswith(self._RAW_SUFFIX)]
        indexCols = [c for c in self.fullPivotDf.columns
                     if c not in predCols and c not in rawCols and c not in sysPromptCols]

        labelCols = [c for c in ('originalLabel', 'trueLabel') if c in indexCols]
        idCols = [c for c in ('itemID',) if c in indexCols]
        otherIndexCols = [c for c in indexCols if c not in labelCols and c not in idCols]

        orderedCols = idCols + labelCols + rawCols + predCols + sysPromptCols + otherIndexCols
        return self.fullPivotDf[orderedCols]

    def _logRunStats(self):
        """完成 log：兩個 DataFrame 的 shape 與本批 parse 成功率。"""
        validCount = (self.inputDf['predLabel'] != -1).sum()
        totalCount = len(self.inputDf)
        fullShapeStr = str(self.fullPivotDf.shape) if self.fullPivotDf is not None else "N/A"
        logging.info(
            f"[Processor] 完成: partial={self.pivotDf.shape}, full={fullShapeStr}, "
            f"parse rate={validCount}/{totalCount} ({validCount/totalCount:.1%}) "
            f"→ {self.partialInfoCsvPath}"
        )

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _getFeatureCols(self) -> List[str]:
        """回傳 pivot 後新生成的預測欄（不在原始 inputDf 欄位集合內，且不帶 __raw 後綴）。"""
        originalColsSet = set(self.inputDf.columns)
        return [c for c in self.pivotDf.columns
                if c not in originalColsSet and not c.endswith(self._RAW_SUFFIX)]
