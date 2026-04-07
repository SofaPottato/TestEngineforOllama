import pandas as pd
import logging
from pathlib import Path
from .schemas import PipelineError

class LLMResultProcessor:
    def __init__(self, inputCsvPath: Path, outputCsvPath: Path, mergedPath: Path, originalDf: pd.DataFrame):
        """
        初始化資料處理器。
        負責將 OutputParser 產出的長表格（每列一個 pair 的預測結果），
        清理後轉置為寬表格（每列一筆資料、每欄一個模型/prompt 組合的預測值）。

        :param inputCsvPath: OutputParser 產出的結構化 CSV（長表格）
        :param outputCsvPath: 處理後的寬表格 CSV 輸出路徑
        :param mergedPath: 合併原始欄位（Title/Abstract 等）後的完整版 CSV 輸出路徑
        :param originalDf: 原始資料集 DataFrame，用於 merge 補充欄位
        """
        self.inputCsvPath = Path(inputCsvPath)
        self.outputCsvPath = Path(outputCsvPath)
        self.mergedPath = Path(mergedPath)
        self.originalDf = originalDf

        self.requiredColsList = ['Model', 'promptID', 'Pred_Label', 'True_Label']  # 計算評估指標必要欄位
        self.indexColsList = ['Data_ID', 'PMID', 'E1', 'E2']                       # Pivot 時作為行索引的欄位
        self.inputDf = None   # 讀取後的長表格
        self.pivotDf = None   # 轉置後的寬表格

        logging.info(f"LLMResultProcessor(inputCsvPath='{self.inputCsvPath}', outputCsvPath='{self.outputCsvPath}')")

    def doCleanAndMerge(self):
        """
        [Public] 執行完整的處理流程：讀取 → 轉換 True_Label → 建立 Feature_Name → Pivot → 存檔。

        :return: 處理後的寬表格 CSV 路徑（outputCsvPath），傳給 PromptCmbEval 使用
        """
        logging.info(f"process(self=<{self.__module__}.{self.__class__.__name__} object at {hex(id(self))}>)")
        logging.info(f"Processing data: {self.inputCsvPath}")

        self.doLoadData()

        logging.info("Processing True Labels")
        self.inputDf['True_Label'] = self.inputDf['True_Label'].apply(self._doConvertTrueLabel)

        # 轉換後檢查有沒有未知值
        unknownCount = (self.inputDf['True_Label'] == -1).sum()
        if unknownCount > 0:
            logging.warning(f"⚠️ 有 {unknownCount} 筆 True_Label 無法識別，這些樣本將在評估時被自動排除")

        logging.info("Creating Feature Names")
        # Feature_Name 作為 Pivot 後的欄名，格式如 "llama3.2_1b_EMO01"
        self.inputDf['Feature_Name'] = self.inputDf['Model'].astype(str) + "_" + self.inputDf['promptID'].astype(str)

        self.doPivotData()

        return self.doSaveData()

    def doLoadData(self):
        """
        讀取並驗證輸入 CSV，結果存入 self.inputDf。

        :raises PipelineError: 找不到檔案、讀取失敗、或缺少必要欄位時
        """
        logging.info("Loading Raw CSV Data...")

        if not self.inputCsvPath.exists():
            raise PipelineError(f"找不到檔案: {self.inputCsvPath}")

        try:
            self.inputDf = pd.read_csv(self.inputCsvPath)
            logging.info(f"Data loaded successfully. Shape: {self.inputDf.shape}")
        except Exception as e:
            raise PipelineError(f"讀取 CSV 失敗: {e}") from e

        # 同時檢查業務欄位（requiredColsList）與索引欄位（indexColsList），一次報告所有缺失
        missingColsList = [c for c in self.requiredColsList + self.indexColsList if c not in self.inputDf.columns]
        if missingColsList:
            raise PipelineError(f"缺少必要欄位: {missingColsList}")

    def _doConvertTrueLabel(self, x) -> int:
        """
        將原始 True_Label 字串標準化為整數。
        本資料集的正類標籤為 'CID'（Chemical-Induced Disease），負類為 0/false/none/negative。

        :param x: 原始標籤值（字串或數字）
        :return: 1 (正類)、0 (負類)、-1 (無法識別)
        """
        val = str(x).strip().lower()
        if val == 'cid':
            return 1
        elif val in ['0', 'false', 'none', 'negative']:
            return 0
        else:
            logging.warning(f"⚠️ 未預期的 True_Label 值: '{x}'，將標記為 -1")
            return -1  # 未知值標記為 -1，不會被當成負類，評估時會被 validMask 排除

    def doPivotData(self):
        """
        將長表格（每列一個 pair × model × prompt 的預測結果）
        轉置為寬表格（每列一筆資料，每欄一個模型/prompt 組合）。

        Pivot 結構：
          - index：Data_ID / PMID / E1 / E2 / True_Label
          - columns：Feature_Name（如 llama3.2_1b_EMO01）
          - values：Pred_Label（0/1/-1）

        :raises PipelineError: 轉置失敗時（通常是重複的 index + column 組合）
        """
        logging.info("Pivoting table (Long to Wide)")
        try:
            self.pivotDf = self.inputDf.pivot_table(
                index=self.indexColsList + ['True_Label'],
                columns='Feature_Name',
                values='Pred_Label',
                aggfunc='first'  # 若同一組合有重複紀錄，取第一筆（理論上不應出現重複）
            )
            self.pivotDf = self.pivotDf.reset_index()   # 將 index 欄位還原為一般欄位
            self.pivotDf = self.pivotDf.fillna(-1)      # 無預測值（任務跳過或解析失敗）補 -1
            logging.info(f"Pivot completed. New Shape: {self.pivotDf.shape}")
        except Exception as e:
            raise PipelineError(f"表格轉置（pivot）失敗: {e}") from e

    def doSaveData(self) -> Path:
        """
        儲存兩份結果：
        1. 乾淨寬表格（outputCsvPath）：僅含索引欄位 + 各模型預測欄，供 PromptCmbEval 使用
        2. 完整資訊版（mergedPath）：額外 merge 原始 Title/Abstract 等欄位，供人工審閱

        :return: 乾淨寬表格的路徑（outputCsvPath）
        :raises PipelineError: 儲存失敗時
        """
        try:
            self.outputCsvPath.parent.mkdir(parents=True, exist_ok=True)
            self.pivotDf.to_csv(self.outputCsvPath, index=False, encoding='utf-8-sig')

            if self.mergedPath and self.originalDf is not None:
                logging.info("Generating rich merged table for human review...")
                self.mergedPath.parent.mkdir(parents=True, exist_ok=True)

                # 從原始 DataFrame 提取可用的補充欄位（不強制要求所有欄位都存在）
                columnsToAddList = ['Title', 'Abstract', 'Full_Text', 'E1_Type', 'E1_MeSH', 'E2_Type', 'E2_MeSH']
                validColsList = [c for c in columnsToAddList if c in self.originalDf.columns]

                origSubsetDf = self.originalDf[validColsList].copy()
                origSubsetDf['Data_ID'] = origSubsetDf.index  # 以 DataFrame 索引作為 join key

                mergeDf = pd.merge(self.pivotDf, origSubsetDf, on='Data_ID', how='left')

                # 將人工閱讀時最有用的欄位排到最前面，其餘（預測欄）放後面
                frontColsList = ['Data_ID', 'PMID', 'E1', 'E1_Type', 'E2', 'E2_Type',
                                 'True_Label', 'Title', 'Abstract']
                frontColsList = [c for c in frontColsList if c in mergeDf.columns]
                predColsList = [c for c in mergeDf.columns if c not in frontColsList]
                mergeDf = mergeDf[frontColsList + predColsList]

                mergeDf.to_csv(self.mergedPath, index=False, encoding='utf-8-sig')
                logging.info(f"   -資訊總成已儲存至: {self.mergedPath}")

            validCount = (self.inputDf['Pred_Label'] != -1).sum()   # Pred_Label != -1 代表解析成功
            totalCount = len(self.inputDf)

            logging.info("✅ Data processed successfully!")
            logging.info(f"   - Clean Shape: {self.pivotDf.shape}")
            logging.info(f"   - Parse Success Rate: {validCount}/{totalCount} ({validCount/totalCount:.1%})")
            logging.info(f"   - Clean Pipeline Data Saved to: {self.outputCsvPath}")

            return self.outputCsvPath

        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"儲存結果失敗: {e}") from e
