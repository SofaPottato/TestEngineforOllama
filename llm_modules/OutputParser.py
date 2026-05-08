import pandas as pd
import json
import logging
import re
from pathlib import Path
from typing import List, Optional
from .schemas import ParsingError, LabelMapConfig, RESERVED_PAIR_FIELDS
from .utils import sanitizeFilename


class OutputParser:
    """
    將 LLM 純文字回應解析為結構化資料表。
    raw.csv → result.csv（predLabel 0/1/-1）。
    single / batch 模式共用同一組關鍵字（來自 labelMapConfig.outputPositive / outputNegative），
    無法判定一律標 -1，由下游評估時排除。
    """

    # 切割 LLM 批次回應的行首編號（如 "1:", "No. 2", "3)"）
    _NUMBERED_LINE_RE = re.compile(
        r'\n\s*(?:\s+|No\.?\s*)?\d+\s*[:.)-]',
        flags=re.IGNORECASE
    )

    _CSV_KWARGS = {'index': False, 'encoding': 'utf-8-sig'}

    def __init__(self, rawOutputCsvPath: Path, parsedOutputCsvPath: Path, singlePromptCmbOutputDir: Path,
                 labelMapConfig: Optional[LabelMapConfig] = None):
        self.rawOutputCsvPath = Path(rawOutputCsvPath)
        self.parsedOutputCsvPath = Path(parsedOutputCsvPath)
        self.singlePromptCmbOutputDir = Path(singlePromptCmbOutputDir)
        self.labelMapConfig = labelMapConfig or LabelMapConfig()
        self._posKeywords = [k.lower() for k in self.labelMapConfig.outputPositive]
        self._negKeywords = [k.lower() for k in self.labelMapConfig.outputNegative]

    def run(self) -> Path:
        """主流程：讀 raw.csv → 逐 task 解析 → pair 展開（long format）→ 排序 → 存檔。"""
        try:
            rawDf = self._loadRawCsv()
            parsedRowsList = self._parseAllTasks(rawDf)
            parsedDf = self._buildResultDf(parsedRowsList)
            self._writeOutputs(parsedDf)
            logging.info(f"[Parser] 解析完成: {len(parsedDf)} 筆 → {self.parsedOutputCsvPath}")
            return self.parsedOutputCsvPath
        except ParsingError:
            raise
        except Exception as e:
            raise ParsingError(f"解析暫存檔時發生錯誤: {e}") from e

    # ── 私有流程方法 ─────────────────────────────────────────────────────────

    def _loadRawCsv(self) -> pd.DataFrame:
        if not self.rawOutputCsvPath.exists():
            raise ParsingError(f"找不到暫存結果檔案: {self.rawOutputCsvPath}")
        return pd.read_csv(str(self.rawOutputCsvPath), encoding='utf-8-sig')

    def _parseAllTasks(self, rawDf: pd.DataFrame) -> list:
        hasContextCol = 'context' in rawDf.columns
        parsedRowsList = []
        for _, taskRow in rawDf.iterrows():
            parsedRowsList.extend(self._parseTaskRow(taskRow, hasContextCol))
        return parsedRowsList

    def _parseTaskRow(self, taskRow, hasContextCol: bool) -> list:
        """一列 task row → 展開成多列 rowDict（每個 pair 一列）。"""
        model    = taskRow.get('model')
        promptID = taskRow.get('promptID')
        taskID   = str(taskRow.get('taskID', ''))

        pairsList = self._parseJsonCell(taskRow.get('pairs'), default=[])
        if not pairsList:
            logging.warning(f"[Parser] 跳過任務: pairs 為空 (model={model}, promptID={promptID})")
            return []

        rawOutput   = str(taskRow.get('rawOutput', ''))
        answers     = self._extractAnswers(rawOutput, len(pairsList))
        contextDict = (self._parseJsonCell(taskRow.get('context'), default={})
                       if hasContextCol else {})

        return [
            self._buildRow(model, promptID, taskID, rawOutput,
                           pairDict, answers[j], j, len(pairsList), contextDict)
            for j, pairDict in enumerate(pairsList)
        ]

    def _buildRow(self, model, promptID, taskID, rawOutput,
                  pairDict: dict, predLabel: int,
                  pairIndex: int, totalPairs: int, contextDict: dict) -> dict:
        """單一 pair + 對應預測標籤 → rowDict。"""
        itemID = pairDict.get('itemID') or (f"{taskID}_{pairIndex}" if totalPairs > 1 else taskID)
        rowDict = {
            "itemID":    itemID,
            "model":     model,
            "promptID":  promptID,
            "trueLabel": pairDict.get('label', ''),
            "predLabel": predLabel,
            "rawOutput": rawOutput,
        }
        # pair 中非保留欄位（id, e1, e2 等）全部帶入
        for fieldName, fieldVal in pairDict.items():
            if fieldName not in RESERVED_PAIR_FIELDS:
                rowDict[fieldName] = fieldVal
        # context 欄位補充（pair 優先，不覆蓋）
        for fieldName, fieldVal in contextDict.items():
            if fieldName not in rowDict:
                rowDict[fieldName] = fieldVal
        return rowDict

    def _buildResultDf(self, parsedRowsList: list) -> pd.DataFrame:
        if not parsedRowsList:
            raise ParsingError("解析後沒有產生任何有效資料。")
        parsedDf = pd.DataFrame(parsedRowsList)
        return parsedDf.sort_values(['model', 'promptID', 'itemID'])

    def _writeOutputs(self, parsedDf: pd.DataFrame) -> None:
        """輸出合併版 result.csv，同時按 promptID 分檔。"""
        parsedDf.to_csv(str(self.parsedOutputCsvPath), **self._CSV_KWARGS)
        for promptID, groupDf in parsedDf.groupby('promptID'):
            singleCsvPath = self.singlePromptCmbOutputDir / f"{sanitizeFilename(promptID)}_result.csv"
            groupDf.to_csv(singleCsvPath, **self._CSV_KWARGS)

    # ── 解析工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _parseJsonCell(rawValue, default):
        """
        寬鬆解析 raw.csv 的 JSON 欄位（pairs / context）。
        NaN/None/非法型別 → default；字串 → json.loads（失敗記 warning 後 → default）。
        """
        if rawValue is None or (isinstance(rawValue, float) and pd.isna(rawValue)):
            return default
        if isinstance(rawValue, (dict, list)):
            return rawValue
        if isinstance(rawValue, str):
            stripped = rawValue.strip()
            if not stripped:
                return default
            try:
                return json.loads(stripped)
            except Exception as e:
                logging.warning(f"[Parser] JSON 欄位解析失敗，回傳 default: {e}")
                return default
        return default

    def _extractAnswers(self, text: str, batchSize: int) -> List[int]:
        """
        從 LLM 回應切分出每一題的答案，回傳長度為 batchSize 的 list（1/0/-1）。
        "Error:" 開頭 → 全部 -1；batchSize > 1 → 以行首編號 regex 切段再逐段掃描。
        """
        labelResultsList = [-1] * batchSize
        if not text or "Error:" in text:
            return labelResultsList

        text = text.replace('*', '')  # 移除 Markdown 粗體標記，避免干擾關鍵字掃描

        if batchSize == 1:
            labelResultsList[0] = self._scanBlock(text.strip())
            return labelResultsList

        blocks = self._NUMBERED_LINE_RE.split("\n" + text.strip())[1:]
        for i in range(batchSize):
            if i < len(blocks):
                labelResultsList[i] = self._scanBlock(blocks[i])

        return labelResultsList

    def _scanBlock(self, blockText: str) -> int:
        """
        Substring 關鍵字掃描，回傳 1 / 0 / -1。
        正類優先：同時出現正負關鍵字時（如 "yes, but no..."），通常 yes 才是主要答案。
        """
        loweredText = blockText.lower()
        if any(k in loweredText for k in self._posKeywords):
            return 1
        if any(k in loweredText for k in self._negKeywords):
            return 0
        return -1
