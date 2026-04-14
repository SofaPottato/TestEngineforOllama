import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from .OllamaEngine import LLMEngine
from .OutputParser import OutputParser
from .LLMResultProcessor import LLMResultProcessor
from .Evaluate import PromptCmbEval
from .PromptRenderer import PromptRenderer
from .schemas import DataLoadError, TaskBuildError, InferenceError, LLMAppConfig, LLMTask


class ExperimentPipeline:
    def __init__(self, config: LLMAppConfig):
        """
        實驗流程的統籌中心。
        讀取前處理產出的標準 Task CSV，搭配 config 中的 template 渲染 prompt，
        再交給 LLM 推論、解析、評估。

        :param config: 通過 Pydantic 驗證的完整設定物件 (LLMAppConfig)
        """
        logging.info("Initializing ExperimentPipeline()")
        self.config = config
        p = config.paths

        self.taskCsvPath = p.taskCsvPath
        self.promptCmbPath = p.promptCmbPath
        self.rawOutputPath = p.rawOutputPath
        self.resultPath = p.resultPath
        self.singlePromptOutputDir = p.singlePromptOutputDir
        self.partialInfoPath = p.partialInfoPath
        self.fullInfoPath = p.fullInfoPath
        self.evalDir = p.evalDir

        self.rawOutputPath.parent.mkdir(parents=True, exist_ok=True)
        self.fullInfoPath.parent.mkdir(parents=True, exist_ok=True)

        self.renderer = PromptRenderer(config.taskTemplate, config.itemTemplate)

        logging.info("ExperimentPipeline initialized.")

    # ===================== 資料載入 =====================

    def doLoadTaskCsv(self) -> pd.DataFrame:
        """
        載入前處理產出的標準 Task CSV。
        必要欄位：taskID, context (JSON), items (JSON)

        :return: Task DataFrame
        :raises DataLoadError: 找不到檔案或缺少必要欄位時
        """
        if not self.taskCsvPath.exists():
            raise DataLoadError(f"找不到 Task CSV: {self.taskCsvPath}")

        taskDf = pd.read_csv(self.taskCsvPath, encoding='utf-8-sig')
        required = {'taskID', 'context', 'items'}
        missing = required - set(taskDf.columns)
        if missing:
            raise DataLoadError(f"Task CSV 缺少必要欄位: {missing}")

        logging.info(f"Task CSV loaded: {len(taskDf)} tasks from {self.taskCsvPath}")
        return taskDf

    def doLoadPromptCmb(self) -> List[Dict[str, str]]:
        """
        載入 Prompt 組合 CSV，回傳 List[Dict]。
        每個 Dict 含 'promptID' 與 'promptText'。

        :return: List[{'promptID': str, 'promptText': str}]
        :raises DataLoadError: 找不到檔案或缺少必要欄位時
        """
        if not self.promptCmbPath.exists():
            raise DataLoadError(f"找不到 Prompt 組合檔案: {self.promptCmbPath}")

        promptDf = pd.read_csv(self.promptCmbPath, encoding='utf-8-sig')
        if 'promptID' not in promptDf.columns or 'promptText' not in promptDf.columns:
            raise DataLoadError("Prompt CSV 缺少 'promptID' 或 'promptText' 欄位。")

        return promptDf[['promptID', 'promptText']].to_dict(orient='records')

    # ===================== 任務建構 =====================

    def doBuildLLMTasks(self, taskDf: pd.DataFrame, promptCmbList: List[Dict],
                        completedIDs: set) -> List[LLMTask]:
        """
        將 Task CSV × models × prompts 排列組合，產出 LLMTask list。
        已完成的 taskID 會被跳過（斷點續傳）。

        :param taskDf: 前處理產出的 Task DataFrame
        :param promptCmbList: Prompt 組合清單
        :param completedIDs: 已完成任務的 taskID set
        :return: 待執行的 LLMTask list
        """
        tasksToRun: List[LLMTask] = []
        skipped = 0

        for model in self.config.selectedModels:
            for promptDict in promptCmbList:
                promptID = promptDict['promptID']
                sysPrompt = promptDict['promptText']

                for _, row in taskDf.iterrows():
                    taskBaseID = str(row['taskID'])
                    context = json.loads(row['context']) if isinstance(row['context'], str) else row['context']
                    items = json.loads(row['items']) if isinstance(row['items'], str) else row['items']

                    fullTaskID = f"{model}::{promptID}::{taskBaseID}"

                    if fullTaskID in completedIDs:
                        skipped += 1
                        continue

                    userPrompt = self.renderer.render(context, items)

                    task = LLMTask(
                        taskID=fullTaskID,
                        model=model,
                        promptID=promptID,
                        sysPrompt=sysPrompt,
                        userPrompt=userPrompt,
                        items=items
                    )
                    tasksToRun.append(task)

        if skipped > 0:
            logging.info(f"Skipped {skipped} previously completed tasks.")
        logging.info(f"Built {len(tasksToRun)} new tasks to run.")
        return tasksToRun

    # ===================== 斷點續傳 =====================

    def doGetCompletedTasks(self) -> set:
        """
        讀取推論暫存檔，取得所有已完成任務的 taskID，用於斷點續傳。

        :return: 已完成任務的 taskID set
        """
        completedIDs = set()
        if not self.rawOutputPath.exists():
            return completedIDs

        try:
            checkpointDf = pd.read_csv(str(self.rawOutputPath), encoding='utf-8-sig')
            if 'taskID' in checkpointDf.columns:
                completedIDs.update(checkpointDf['taskID'].dropna().astype(str).str.strip().tolist())
            else:
                logging.warning("Checkpoint file missing 'taskID' column.")
            logging.info(f"Checkpoint loaded: {len(completedIDs)} completed tasks.")
        except Exception as e:
            logging.warning(f"Failed to read checkpoint, starting fresh: {e}")

        return completedIDs

    # ===================== 推論 =====================

    def doRunInference(self, tasksToRun: List[LLMTask], completedIDs: set):
        """
        建立 LLMEngine 並以非同步方式執行所有推論任務。

        :param tasksToRun: 尚未完成的 LLMTask 物件清單
        :param completedIDs: 已完成任務的 taskID set
        """
        engine = LLMEngine(
            apiUrl=self.config.ollamaServer.url,
            timeout=self.config.ollamaServer.timeout,
            llmOptions=self.config.llmOptions,
            concurrencyPerModel=self.config.concurrencyPerModel,
            maxConcurrentModels=self.config.maxConcurrentModels,
            outputFile=str(self.rawOutputPath),
            existingTaskIds=completedIDs
        )

        logging.info(f"Dispatching {len(tasksToRun)} tasks to async engine...")
        taskDictList = [task.model_dump() for task in tasksToRun]
        asyncio.run(engine.doExecuteTaskBatches(taskDictList))
        logging.info(f"Inference complete. Raw output saved to: {self.rawOutputPath}")

    # ===================== 主流程 =====================

    def run(self):
        """
        實驗主流程：
          1. 載入 Task CSV 與 Prompt 組合
          2. 建構推論任務（支援斷點續傳）
          3. 執行 LLM 推論
          4. 解析 LLM 原始回應
          5. 清理資料、轉置為寬格式
          6. 評估各模型/prompt 組合的分類效能
        """
        logging.info("==== [Step 1] Loading Task CSV & Prompts ====")
        taskDf = self.doLoadTaskCsv()
        promptCmbList = self.doLoadPromptCmb()

        logging.info("==== [Step 2] Building LLM Tasks ====")
        completedIDs = self.doGetCompletedTasks()
        tasksToRun = self.doBuildLLMTasks(taskDf, promptCmbList, completedIDs)

        if not tasksToRun and not completedIDs:
            raise TaskBuildError("No tasks to run and no checkpoint found.")

        if tasksToRun:
            logging.info(f"==== [Step 3] Running Inference ({len(tasksToRun)} tasks) ====")
            try:
                self.doRunInference(tasksToRun, completedIDs)
            except Exception as e:
                raise InferenceError(f"Inference failed: {e}") from e
        else:
            logging.info("==== [Step 3] All tasks completed. Skipping inference. ====")

        logging.info("==== [Step 4] Parsing LLM Outputs ====")
        parser = OutputParser(
            rawCsvPath=self.rawOutputPath,
            csvOutputPath=self.resultPath,
            singlePromptCmbOutputDir=self.singlePromptOutputDir
        )
        parsedPath = parser.doParse()

        logging.info("==== [Step 5] Processing Results ====")
        processor = LLMResultProcessor(
            inputCsvPath=parsedPath,
            outputCsvPath=self.partialInfoPath,
            mergedPath=self.fullInfoPath,
            labelMap=self.config.labelMap
        )
        processedPath = processor.doCleanAndMerge()

        logging.info("==== [Step 6] Evaluating ====")
        evaluator = PromptCmbEval(
            inputCsvPath=processedPath,
            outputBaseDir=self.evalDir
        )
        evaluator.doEval()
        evaluator.doAnalyzeUpperBound()
        evaluator.doPlotConfusionMatrices()
        evaluator.doPlotHeatmap()
        evaluator.doSaveResults()

        logging.info("Pipeline completed successfully!")
