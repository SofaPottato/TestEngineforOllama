import pandas as pd
import json
import logging
import re
from pathlib import Path
from typing import List
from .schemas import ParsingError


class OutputParser:
    def __init__(self, rawCsvPath: Path, outputCsvPath: Path, singlePromptCmbOutputDir: Path):
        """
        初始化輸出解析器。
        負責將 LLM 的 Raw Output (字串) 透過 Regex 拆解成結構化的 DataFrame。

        :param rawCsvPath: 推論暫存 CSV 的路徑（由 LLMEngine 逐筆寫入）
        :param outputCsvPath: 解析後統整 CSV 的輸出路徑
        :param singlePromptCmbOutputDir: 每個 promptID 單獨存檔的目錄
        """
        self.rawCsvPath = Path(rawCsvPath)
        self.outputCsvPath = Path(outputCsvPath)
        self.singlePromptCmbOutputDir = Path(singlePromptCmbOutputDir)
        logging.info("OutputParser Initialized.")

    def doExtractAnswers(self, text: str, batchSize: int) -> List[int]:
        """
        核心解析邏輯：從 LLM 的文字回應中，切分出每一題的答案。

        當 batchSize == 1 時（單筆模式），直接對整段回應做關鍵字掃描。
        當 batchSize > 1 時（批次模式），以編號切分後逐段掃描。

        :param text: LLM 的原始回應字串
        :param batchSize: 本次批次應有的答案數量
        :return: 長度為 batchSize 的整數 List，值為 1/0/-1
        """
        labelResultsList = [-1] * batchSize
        if not text or "Error:" in text:
            return labelResultsList

        text = text.replace('*', '')

        if batchSize == 1:
            blockText = text.lower().strip()
            if 'yes' in blockText or 'positive' in blockText:
                labelResultsList[0] = 1
            elif 'no' in blockText or 'negative' in blockText or 'none' in blockText:
                labelResultsList[0] = 0
            return labelResultsList

        # 批次模式：以編號切分
        text = "\n" + text.strip()
        blocksList = re.split(r'\n\s*(?:\s+|No\.?\s*)?\d+\s*[:.)-]', text, flags=re.IGNORECASE)
        blocksList = blocksList[1:]

        for i in range(batchSize):
            if i < len(blocksList):
                blockText = blocksList[i].lower()
                if 'yes' in blockText:
                    labelResultsList[i] = 1
                elif 'no' in blockText or 'none' in blockText:
                    labelResultsList[i] = 0

        return labelResultsList

    def run(self) -> Path:
        """
        讀取推論暫存 CSV，對每一筆任務套用解析，
        將 pairs 展開為逐筆資料列，輸出結構化 CSV。

        Raw CSV 欄位：taskID, model, promptID, rawOutput, pairs (JSON array)
        每個 pair 必帶 id, label；其餘為自訂欄位。

        輸出欄位：dataID, Model, promptID, trueLabel, predLabel, rawOutput, [自訂欄位...]

        :return: 解析後 CSV 的路徑
        :raises ParsingError: 找不到暫存檔、或解析後無有效資料時
        """
        logging.info("==== [OutputParser] Parsing LLM Outputs & Building CSV ====")
        try:
            if not self.rawCsvPath.exists():
                raise ParsingError(f"找不到暫存結果檔案: {self.rawCsvPath}")

            rawDf = pd.read_csv(str(self.rawCsvPath), encoding='utf-8-sig')
            parsedRowsList = []

            hasContextCol = 'context' in rawDf.columns

            for _, taskRow in rawDf.iterrows():
                model = taskRow.get('model')
                promptID = taskRow.get('promptID')
                rawOutput = str(taskRow.get('rawOutput', ''))

                # 讀取 pairs JSON
                pairsRaw = taskRow.get('pairs', '[]')
                if pd.isna(pairsRaw):
                    pairsRaw = '[]'

                try:
                    if isinstance(pairsRaw, str):
                        pairsList = json.loads(pairsRaw)
                    elif isinstance(pairsRaw, list):
                        pairsList = pairsRaw
                    else:
                        pairsList = []
                except Exception as e:
                    logging.warning(f"Failed to parse pairs JSON: {e}")
                    pairsList = []

                # 讀取 context JSON（task 層級欄位：title/abstract/passage 等）
                contextDict: dict = {}
                if hasContextCol:
                    contextRaw = taskRow.get('context', '{}')
                    if not pd.isna(contextRaw):
                        try:
                            if isinstance(contextRaw, str) and contextRaw.strip():
                                contextDict = json.loads(contextRaw)
                            elif isinstance(contextRaw, dict):
                                contextDict = contextRaw
                        except Exception as e:
                            logging.warning(f"Failed to parse context JSON: {e}")

                if not pairsList:
                    logging.error(f"Empty pairs for task (Model: {model}, Prompt: {promptID})")
                    continue

                parsedAnswersList = self.doExtractAnswers(rawOutput, len(pairsList))

                # 當 pair 沒帶 id 時（例如 PPI 單句資料集），用 taskID 最後一段當 fallback
                rawTaskID = str(taskRow.get('taskID', ''))
                baseTaskID = rawTaskID.split('::')[-1]

                for j, pair in enumerate(pairsList):
                    predLabel = parsedAnswersList[j] if j < len(parsedAnswersList) else -1

                    fallbackID = f"{baseTaskID}_{j}" if len(pairsList) > 1 else baseTaskID
                    rowDict = {
                        "dataID": pair.get('id') or fallbackID,
                        "Model": model,
                        "promptID": promptID,
                        "trueLabel": pair.get('label', ''),
                        "predLabel": predLabel,
                        "rawOutput": rawOutput
                    }

                    # 自訂欄位展開（id/label 以外的欄位）
                    for fieldName, fieldVal in pair.items():
                        if fieldName not in ('id', 'label'):
                            rowDict[fieldName] = fieldVal

                    # Task 層級 context 欄位（避免覆蓋 pair 欄位）
                    for fieldName, fieldVal in contextDict.items():
                        if fieldName not in rowDict:
                            rowDict[fieldName] = fieldVal

                    parsedRowsList.append(rowDict)

            parsedDf = pd.DataFrame(parsedRowsList)

            if parsedDf.empty:
                raise ParsingError("解析後沒有產生任何有效資料。")

            parsedDf = parsedDf.sort_values(['Model', 'promptID', 'dataID'])
            parsedDf.to_csv(str(self.outputCsvPath), index=False, encoding='utf-8-sig')

            for promptID, groupDf in parsedDf.groupby('promptID'):
                safeNameStr = str(promptID).replace(":", "_").replace("+", "_").replace(" ", "_").replace("/", "_")
                singlePath = self.singlePromptCmbOutputDir / f"{safeNameStr}_result.csv"
                groupDf.to_csv(singlePath, index=False, encoding='utf-8-sig')

            logging.info(f"Parsing complete: {len(parsedDf)} records -> {self.outputCsvPath}")
            return self.outputCsvPath

        except ParsingError:
            raise
        except Exception as e:
            raise ParsingError(f"解析暫存檔時發生錯誤: {e}") from e
