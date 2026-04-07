import asyncio
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from .TaskBuilder import TaskBuilder
from .OllamaEngine import LLMEngine
from .OutputParser import OutputParser
from .LLMResultProcessor import LLMResultProcessor
from .Evaluate import PromptCmbEval
from .schemas import DataLoadError, TaskBuildError, InferenceError, LLMAppConfig


class ExperimentPipeline:
    def __init__(self, config: LLMAppConfig):
        """
        實驗流程的統籌中心。
        負責接收設定檔、初始化所有路徑，並在啟動時建立必要的輸出資料夾。
        實際的執行邏輯統一交由 run() 調度。

        :param config: 通過 Pydantic 驗證的完整設定物件 (LLMAppConfig)
        """
        logging.info("Initializing ExperimentPipeline()")
        self.configObj = config

        pathsConfig = self.configObj.paths
        self.dataPath = pathsConfig.dataPath                             # 原始資料集 CSV（含 E1/E2 pair 與標籤）
        self.promptCmbPath = pathsConfig.promptCmbPath                   # Prompt 組合 CSV（promptID + promptText）
        self.rawPromptCmbOutputPath = pathsConfig.rawPromptCmbOutputPath # LLM 推論原始暫存 CSV（含 rawOutput），斷點續傳的依據
        self.allPromptsCmbResultPath = pathsConfig.allPromptsCmbResultPath   # OutputParser 解析後的結構化 CSV（Pred_Label 已逐筆展開）
        self.promptCmbPartialInfoPath = pathsConfig.promptCmbPartialInfoPath # Pivot 後的寬格式 CSV，每列一筆資料、每欄一個模型/prompt 組合
        self.promptCmbFullInfoPath = pathsConfig.promptCmbFullInfoPath       # 同上，額外合併 Title/Abstract 等原始欄位，供人工審閱
        self.singlePromptCmbOutputDir = pathsConfig.singlePromptCmbOutputDir # 每個 promptID 獨立存一份 CSV 的目錄
        self.promptCmbEvalDir = pathsConfig.promptCmbEvalDir                 # 評估圖表與指標報表的輸出目錄

        # 'Dir' 字尾的路徑已由 PathsConfig 的 field_validator 自動建立
        self.rawPromptCmbOutputPath.parent.mkdir(parents=True, exist_ok=True)
        self.promptCmbFullInfoPath.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"ExperimentPipeline initialized.")

    def doGetCompletedTasks(self) -> set:
        """
        讀取推論暫存檔，取得所有已完成任務的 taskID，用於斷點續傳。

        taskID 格式：{model}::{promptID}::{firstPairData_ID}
        若暫存檔不存在（首次執行），回傳空 set，全部任務重新跑。
        若暫存檔讀取失敗（損壞等），也回傳空 set 並記錄警告，確保 pipeline 不中斷。

        :return: 已完成任務的 taskID set
        """
        completedIDSet = set()

        # 暫存檔不存在代表首次執行，直接回傳空 set
        if not self.rawPromptCmbOutputPath.exists():
            return completedIDSet

        try:
            checkpointDf = pd.read_csv(str(self.rawPromptCmbOutputPath), encoding='utf-8-sig')

            if 'taskID' in checkpointDf.columns:
                # dropna: 排除空值；astype(str).str.strip(): 確保型別一致、去除空白
                # 避免因格式差異（如數字 ID、前後空格）導致比對失敗、重複執行同一任務
                completedIDSet.update(checkpointDf['taskID'].dropna().astype(str).str.strip().tolist())
            else:
                logging.warning("⚠️ 發現的暫存檔缺少 taskID，無法使用斷點續傳。")

            logging.info(f"發現斷點紀錄！已載入 {len(completedIDSet)} 筆完成的任務身分證。")
        except Exception as e:
            logging.warning(f"⚠️ 讀取斷點紀錄失敗，將忽略舊紀錄重新開始: {e}")

        return completedIDSet

    def doRunInference(self, tasksToRunList: List, completedIDSet: set):
        """
        建立 LLMEngine 並以非同步方式執行所有推論任務。
        結果由 LLMEngine 內部逐筆寫入 CSV，不透過回傳值傳遞。
        :param tasksToRunList: 尚未完成的 LLMTask 物件清單
        :param completedIDSet: 已完成任務的 taskID set（用於引擎內部的重複檢查）
        """
        if self.rawPromptCmbOutputPath.exists():
            logging.info(f"發現既存的暫存檔，引擎將自動接續寫入: {self.rawPromptCmbOutputPath}")
        else:
            logging.info(f"建立全新暫存檔: {self.rawPromptCmbOutputPath}")

        llmEngineObj = LLMEngine(
            apiUrl=self.configObj.ollamaServer.url,
            timeout=self.configObj.ollamaServer.timeout,
            llmOptions=self.configObj.llmOptions,
            concurrencyPerModel=self.configObj.concurrencyPerModel,
            maxConcurrentModels=self.configObj.maxConcurrentModels,
            outputFile=str(self.rawPromptCmbOutputPath),
            existingTaskIds=completedIDSet
        )

        logging.info(f"交接給非同步引擎執行 (總任務批次數: {len(tasksToRunList)})...")

        # LLMEngine 接受 dict 格式，將 Pydantic 物件序列化後傳入
        taskDictList = [task.model_dump() for task in tasksToRunList]

        # 僅用於確認引擎是否有回傳結果，實際輸出已由引擎寫入 CSV
        inferenceResultsList = asyncio.run(llmEngineObj.doExecuteTaskBatches(taskDictList))

        if not inferenceResultsList:
            logging.error("❌ 推論引擎回傳空結果。")

        logging.info(f"✅ 推論完成！原始回應 (Raw Output) 已儲存至: {self.rawPromptCmbOutputPath}")

    def run(self):
        """
        實驗主流程，依序執行以下六個步驟：
          1. 載入資料集與 Prompt 組合
          2. 建構推論任務（支援斷點續傳，已完成的任務自動跳過）
          3. 執行 LLM 推論（若所有任務已完成則跳過）
          4. 解析 LLM 原始回應
          5. 清理資料、轉置為寬格式、合併原始資訊
          6. 評估各模型/prompt 組合的分類效能並輸出報表
        """
        logging.info(f"==== [Step 1] Loading Data & Prompts ====")
        dataDf = self.doLoadDataSet()
        promptCmbList = self.doLoadPromptCmb(self.promptCmbPath)

        logging.info("==== [Step 2] Building Tasks ====")
        # 先取得已完成的任務，再傳入 TaskBuilder 跳過這些任務
        completedIDSet = self.doGetCompletedTasks()
        taskBuilderObj = TaskBuilder(
            models=self.configObj.selectedModels,
            pairBatchEnabled=self.configObj.pairSettings.enabled,
            pairNumber=self.configObj.pairSettings.pairNumbers,
            taskTemplate=self.configObj.taskTemplate
        )

        tasksToRunList = taskBuilderObj.doBuildLLMTasks(dataDf, promptCmbList, completedTasks=completedIDSet)

        # 兩個條件都成立才是真正的錯誤：
        # - 沒有新任務（tasksToRunList 為空）
        # - 也沒有歷史紀錄（completedIDSet 為空）
        # → 代表資料或 prompt 設定有問題，無法繼續
        if not tasksToRunList and not completedIDSet:
            raise TaskBuildError("無效的任務建構：無法生成任何新任務，且沒有找到歷史斷點紀錄。")

        if tasksToRunList:
            logging.info(f"==== [Step 3] Running LLM Inference ({len(tasksToRunList)} tasks remaining) ====")
            try:
                self.doRunInference(tasksToRunList, completedIDSet)
            except Exception as e:
                raise InferenceError(f"推論失敗，模型連線異常: {e}") from e
        else:
            # tasksToRunList 為空但 completedIDSet 不為空：所有任務已完成，直接跳到解析
            logging.info("==== [Step 3] All tasks already completed! Skipping Inference. ====")

        logging.info("==== [Step 4] Parsing LLM Outputs ====")
        outputParserObj = OutputParser(
            rawCsvPath=self.rawPromptCmbOutputPath,
            csvOutputPath=self.allPromptsCmbResultPath,
            singlePromptCmbOutputDir=self.singlePromptCmbOutputDir
        )
        parsedOutputPath = outputParserObj.doParse()

        logging.info("==== [Step 5] Processing Results ====")
        resultProcessorObj = LLMResultProcessor(
            inputCsvPath=parsedOutputPath,
            outputCsvPath=self.promptCmbPartialInfoPath,
            mergedPath=self.promptCmbFullInfoPath,
            originalDf=dataDf     # 原始 DataFrame，用於合併 Title/Abstract 等欄位
        )
        processedDataPath = resultProcessorObj.doCleanAndMerge()

        logging.info("==== [Step 6] Evaluating ====")
        evalObj = PromptCmbEval(
            inputCsvPath=processedDataPath,
            outputBaseDir=self.promptCmbEvalDir
        )

        evalObj.doEval()
        evalObj.doAnalyzeUpperBound()
        evalObj.doPlotConfusionMatrices()
        evalObj.doPlotHeatmap()
        evalObj.doSaveResults()

        logging.info("實驗全部執行完畢！(Pipeline Completed Successfully!)")

    def doLoadDataSet(self) -> pd.DataFrame:
        """
        載入原始資料集 CSV，回傳 DataFrame。
        若設定了 testLimits，僅取前 N 筆，用於開發期間快速驗證流程。

        :return: 原始資料集 DataFrame
        :raises DataLoadError: 找不到檔案時
        """
        if not self.dataPath or not self.dataPath.exists():
            raise DataLoadError(f"找不到資料集檔案: {self.dataPath}")

        dataDf = pd.read_csv(
            self.dataPath,
            encoding='utf-8-sig',
            on_bad_lines='warn'   
        )
        if self.configObj.testLimits is not None:
            dataDf = dataDf.head(self.configObj.testLimits)
            logging.warning(f"⚠️ test: Using only first {self.configObj.testLimits} pairs.")
        return dataDf

    def doLoadPromptCmb(self, path: Path) -> List[Dict[str, str]]:
        """
        載入 Prompt 組合 CSV，回傳 List[Dict]。
        每個 Dict 含 'promptID'（識別碼）與 'promptText'（作為 system prompt 傳給 LLM）。

        :param path: Prompt 組合 CSV 的路徑
        :return: List[{'promptID': str, 'promptText': str}]
        :raises DataLoadError: 找不到檔案或缺少必要欄位時
        """
        if not path.exists():
            raise DataLoadError(f"找不到 Prompt 組合檔案: {path}")

        promptCmbDf = pd.read_csv(path)
        if 'promptID' not in promptCmbDf.columns or 'promptText' not in promptCmbDf.columns:
            raise DataLoadError("Prompt CSV 遺失必要的 'promptID' 或 'promptText' 欄位。")

        return promptCmbDf[['promptID', 'promptText']].to_dict(orient='records')
