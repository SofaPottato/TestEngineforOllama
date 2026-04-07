import pandas as pd
import json
import logging
import re
from pathlib import Path
from typing import List
from .schemas import ParsingError

class OutputParser:
    def __init__(self, rawCsvPath: Path, csvOutputPath: Path, singlePromptCmbOutputDir: Path):
        """
        初始化輸出解析器。
        專門負責將 LLM 的 Raw Output (字串) 透過 Regex 拆解成結構化的 DataFrame。

        :param rawCsvPath: 推論暫存 CSV 的路徑（由 LLMEngine 逐筆寫入）
        :param csvOutputPath: 解析後統整 CSV 的輸出路徑
        :param singlePromptCmbOutputDir: 每個 promptID 單獨存檔的目錄
        """
        self.rawCsvPath = Path(rawCsvPath)
        self.csvOutputPath = Path(csvOutputPath)
        self.singlePromptCmbOutputDir = Path(singlePromptCmbOutputDir)
        self.singlePromptCmbOutputDir.mkdir(parents=True, exist_ok=True)

        logging.info("OutputParser Initialized.")

    def doExtractAnswers(self, text: str, batchSize: int) -> List[int]:
        """
        核心解析邏輯：從 LLM 的文字回應中，精準切分出每一題的答案。

        LLM 的回應格式預期為：
          1. Yes/No
          2. Yes/No
          ...
        或類似的編號列表。對每個區塊進行關鍵字掃描 (yes/cid → 1, no/none → 0)。
        無法辨識的答案保留為 -1（代表解析失敗）。

        :param text: LLM 的原始回應字串
        :param batchSize: 本次批次應有的答案數量（決定回傳 List 的長度）
        :return: 長度為 batchSize 的整數 List，值為 1 (Yes)、0 (No) 或 -1 (解析失敗)
        """
        labelResultsList = [-1] * batchSize  # 預設全部標記為解析失敗，後續成功解析才覆蓋
        if not text or "Error:" in text:      # 空字串或推論階段已記錄的錯誤訊息，直接回傳全 -1
            return labelResultsList

        # Step 1. 把粗體星號清掉（部分 LLM 會用 **Yes** 格式輸出）
        text = text.replace('*', '')

        # Step 2. 在字串最前面補上換行符號，讓第一題也能被 split pattern 正確切割
        text = "\n" + text.strip()

        # Step 3. 以編號分隔符切分文字區塊
        # pattern 必須以 \n 開頭，徹底避免把答案本文中的 "No" 誤判為題號前綴
        # 匹配格式如：\n1. / \n2: / \nNo. 3 / \nItem 5) 等常見變體
        blocksList = re.split(r'\n\s*(?:\s+|No\.?\s*)?\d+\s*[:.)-]', text, flags=re.IGNORECASE)

        # split 後第一個元素是題號之前的開頭文字（通常是空字串），捨棄不用
        blocksList = blocksList[1:]

        for i in range(batchSize):
            if i < len(blocksList):
                blockText = blocksList[i].lower()
                # Fallback 關鍵字掃描：先找 yes/cid，再找 no/none，順序不可顛倒
                # 避免同時出現時（如 "No CID"）誤判
                if 'yes' in blockText or 'cid' in blockText:
                    labelResultsList[i] = 1
                elif 'no' in blockText or 'none' in blockText:
                    labelResultsList[i] = 0

        return labelResultsList

    def doParse(self) -> Path:
        """
        讀取推論暫存 CSV，對每一筆任務套用 Regex 解析，
        將 Batch 展開為逐筆資料列，最終輸出結構化 CSV。

        輸出欄位：Data_ID, PMID, Model, promptID, E1, E2, True_Label, Pred_Label, Raw_Output

        :return: 解析後 CSV 的路徑（csvOutputPath）
        :raises ParsingError: 找不到暫存檔、或解析後無有效資料時
        """
        logging.info("==== [OutputParser] Parsing LLM Outputs & Building CSV ====")
        try:
            if not self.rawCsvPath.exists():
                raise ParsingError(f"找不到暫存結果檔案: {self.rawCsvPath}")

            rawOutputDf = pd.read_csv(str(self.rawCsvPath), encoding='utf-8-sig')
            parsedRowsList = []

            # 遍歷每一筆完成的任務（每筆對應一個 batch，可能包含多個 pair）
            for _, taskResultDict in rawOutputDf.iterrows():
                model = taskResultDict.get('model')
                promptID = taskResultDict.get('promptID')
                rawOutput = str(taskResultDict.get('rawOutput', ''))

                batchDataJsonStr = taskResultDict.get('batchData', '{}')

                if pd.isna(batchDataJsonStr):  # 空值保護，避免 json.loads 收到 float NaN
                    batchDataJsonStr = '{}'

                batchDataDict = {}

                try:
                    if isinstance(batchDataJsonStr, str):
                        batchDataDict = json.loads(batchDataJsonStr)    # 反序列化 JSON 字串
                    elif isinstance(batchDataJsonStr, dict):
                        batchDataDict = batchDataJsonStr                # 已是 dict，直接使用
                except Exception as e:
                    logging.warning(f"⚠️ 解析 batchData 失敗: {e} (原始資料: {batchDataJsonStr})")
                    batchDataDict = {}

                batchPairsList = batchDataDict.get('batchPairsList', [])
                if not batchPairsList:
                    logging.error(f"❌ 警告：這筆任務的 Batch Data 遺失！(Model: {model}, Prompt: {promptID})")

                pmid = batchDataDict.get('pmid', '')

                # 呼叫 Regex 解析，回傳長度與 batchPairsList 相同的答案 List
                parsedAnswersList = self.doExtractAnswers(rawOutput, len(batchPairsList))

                # 將 Batch 展開為原本的一對一資料列 (Row)
                # 每個 pair 獨立成一列，Pred_Label 對應解析結果中的同位置答案
                for j, pairInfoDict in enumerate(batchPairsList):
                    predLabel = parsedAnswersList[j] if j < len(parsedAnswersList) else -1
                    parsedRowsList.append({
                        "Data_ID": pairInfoDict.get('orig_idx', ''),    # DataFrame 原始索引，非 ID 欄位
                        "PMID": pmid,
                        "Model": model,
                        "promptID": promptID,
                        "E1": pairInfoDict.get('E1_Name', ''),
                        "E2": pairInfoDict.get('E2_Name', ''),
                        "True_Label": pairInfoDict.get('True_Label', ''),
                        "Pred_Label": predLabel,
                        "Raw_Output": rawOutput  # 保留原始輸出方便 Debug
                    })

            parsedResultDf = pd.DataFrame(parsedRowsList)

            if parsedResultDf.empty:
                raise ParsingError("解析後沒有產生任何有效資料，無法儲存 CSV。")

            parsedResultDf = parsedResultDf.sort_values(['Model', 'promptID', 'Data_ID'])

            parsedResultDf.to_csv(str(self.csvOutputPath), index=False, encoding='utf-8-sig')

            # 額外依 promptID 分組，將每個 prompt 組合的結果單獨存一份，方便逐一比較
            for promptID, groupDf in parsedResultDf.groupby('promptID'):
                safeFileNameStr = str(promptID).replace(":", "_").replace("+", "_").replace(" ", "_").replace("/", "_")
                singlePromptOutputPath = self.singlePromptCmbOutputDir / f"{safeFileNameStr}_result.csv"
                groupDf.to_csv(singlePromptOutputPath, index=False, encoding='utf-8-sig')

            logging.info(f"✅ 額外儲存完成！單獨 Prompt 結果已放入: {self.singlePromptCmbOutputDir}")
            logging.info(f"✅ 解析完成！格式化資料已儲存至: {self.csvOutputPath}")

            return self.csvOutputPath

        except ParsingError:
            raise
        except Exception as e:
            raise ParsingError(f"解析暫存檔轉最終 CSV 時發生錯誤: {e}") from e
