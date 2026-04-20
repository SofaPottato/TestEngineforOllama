import asyncio
import logging
from typing import List, Dict, Set, Iterator, Tuple
import pandas as pd
from .OllamaEngine import LLMEngine
from .OutputParser import OutputParser
from .LLMResultProcessor import LLMResultProcessor
from .Evaluate import PromptCmbEval
from .PromptFormatter import PromptFormatter
from .utils import parseJsonField
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
        self.paths = config.paths
        self.formatter = PromptFormatter(config.taskTemplate, config.pairTemplate,
                                         config.pairColumns or None)
        logging.info("ExperimentPipeline initialized.")

    # ===================== 資料載入 =====================

    def doLoadTaskCsv(self) -> pd.DataFrame:
        """
        載入前處理產出的標準 Task CSV。
        必要欄位：taskID, pairs (JSON)，以及 contextColumns 宣告的所有欄位（預設 title, abstract）

        :raises DataLoadError: 找不到檔案或缺少必要欄位時
        """
        path = self.paths.taskCsvPath
        if not path.exists():
            raise DataLoadError(f"找不到 Task CSV: {path}")

        taskDf = pd.read_csv(path, encoding='utf-8-sig')
        requiredCols = {'taskID', 'pairs'} | set(self.config.contextColumns)
        missingColsSet = requiredCols - set(taskDf.columns)
        if missingColsSet:
            raise DataLoadError(f"Task CSV 缺少必要欄位: {missingColsSet}")

        logging.info(f"Task CSV loaded: {len(taskDf)} tasks from {path}")
        return taskDf

    def doLoadPromptCmb(self) -> List[Dict[str, str]]:
        """
        載入 Prompt 組合 CSV。

        :return: List[{'promptID': str, 'promptText': str}]
        :raises DataLoadError: 找不到檔案或缺少必要欄位時
        """
        path = self.paths.promptCmbPath
        if not path.exists():
            raise DataLoadError(f"找不到 Prompt 組合檔案: {path}")

        promptDf = pd.read_csv(path, encoding='utf-8-sig')
        if 'promptID' not in promptDf.columns or 'promptText' not in promptDf.columns:
            raise DataLoadError("Prompt CSV 缺少 'promptID' 或 'promptText' 欄位。")

        return promptDf[['promptID', 'promptText']].to_dict(orient='records')

    # ===================== 任務建構 =====================

    def _buildTaskBatches(self, taskDf: pd.DataFrame) -> List[Tuple[str, list, str]]:
        """
        每列預處理一次：parse JSON、依 pairNumber 切片、format userPrompt。
        避免在 model × prompt 迴圈內重複計算。

        :return: [(taskID, pairs, userPrompt), ...]
        """
        pairNumber = self.config.pairNumber
        rowsList = []
        for _, row in taskDf.iterrows():
            taskBaseID = str(row['taskID'])
            context = {f: row[f] for f in self.config.contextColumns}
            allPairs = parseJsonField(row['pairs'], 'pairs', taskBaseID)

            for offset in range(0, len(allPairs), pairNumber):
                batchPairs = allPairs[offset:offset + pairNumber]
                batchID = f"{taskBaseID}_{offset}" if len(allPairs) > pairNumber else taskBaseID
                userPrompt = self.formatter.format(context, batchPairs)
                rowsList.append((batchID, batchPairs, userPrompt))
        return rowsList

    def doSavePromptPreview(self, taskDf: pd.DataFrame, promptCmbList: List[Dict[str, str]]):
        """
        將所有 promptID × task 組合渲染後的 userPrompt 存成 CSV，供人工檢查。
        欄位：taskID, promptID, sysPrompt, userPrompt
        """
        rowsList = self._buildTaskBatches(taskDf)
        records = []
        for prompt in promptCmbList:
            for taskID, _, userPrompt in rowsList:
                records.append({
                    'taskID':     taskID,
                    'promptID':   prompt['promptID'],
                    'sysPrompt':  prompt['promptText'],
                    'userPrompt': userPrompt,
                })

        path = self.paths.promptPreviewPath
        pd.DataFrame(records).to_csv(str(path), index=False, encoding='utf-8-sig')
        logging.info(f"Prompt preview saved: {len(records)} entries -> {path}")

    def _iterPromptCombinations(
        self, promptCmbList: List[Dict[str, str]]
    ) -> Iterator[Tuple[str, str, str]]:
        """Yield (model, promptID, sysPrompt) 的所有組合。"""
        for model in self.config.selectedModels:
            for prompt in promptCmbList:
                yield model, prompt['promptID'], prompt['promptText']

    def doBuildLLMTasks(self, taskDf: pd.DataFrame, promptCmbList: List[Dict[str, str]],
                        completedIDSet: Set[str]) -> List[LLMTask]:
        """
        將 Task CSV × models × prompts 排列組合，產出 LLMTask list。
        已完成的 taskID 會被跳過（斷點續傳）。

        :raises TaskBuildError: 模型或 prompt 清單為空時
        """
        if not self.config.selectedModels:
            raise TaskBuildError("config.selectedModels 為空，無可執行模型。")
        if not promptCmbList:
            raise TaskBuildError("Prompt 組合清單為空。")

        rowsList = self._buildTaskBatches(taskDf)
        tasksToRunList: List[LLMTask] = []
        skippedCount = 0

        for model, promptID, sysPrompt in self._iterPromptCombinations(promptCmbList):
            for taskBaseID, pairsList, userPrompt in rowsList:
                fullTaskID = f"{model}::{promptID}::{taskBaseID}"
                if fullTaskID in completedIDSet:
                    skippedCount += 1
                    continue
                tasksToRunList.append(LLMTask(
                    taskID=fullTaskID,
                    model=model,
                    promptID=promptID,
                    sysPrompt=sysPrompt,
                    userPrompt=userPrompt,
                    pairs=pairsList,
                ))

        if skippedCount > 0:
            logging.info(f"Skipped {skippedCount} previously completed tasks.")
        logging.info(f"Built {len(tasksToRunList)} new tasks to run.")
        return tasksToRunList

    # ===================== 斷點續傳 =====================

    def doGetCompletedTasks(self) -> Set[str]:
        """
        讀取推論暫存檔，取得所有已完成任務的 taskID，用於斷點續傳。
        """
        completedIDSet: Set[str] = set()
        path = self.paths.rawOutputPath
        if not path.exists():
            return completedIDSet

        try:
            checkpointDf = pd.read_csv(path, encoding='utf-8-sig')
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError) as e:
            logging.warning(f"Failed to read checkpoint, starting fresh: {e}")
            return completedIDSet

        if 'taskID' in checkpointDf.columns:
            completedIDSet.update(checkpointDf['taskID'].dropna().astype(str).str.strip().tolist())
        else:
            logging.warning("Checkpoint file missing 'taskID' column.")
        logging.info(f"Checkpoint loaded: {len(completedIDSet)} completed tasks.")

        return completedIDSet

    # ===================== 推論 =====================

    def doRunInference(self, tasksToRunList: List[LLMTask], completedIDSet: Set[str]):
        """
        建立 LLMEngine 並以非同步方式執行所有推論任務。
        """
        engineObj = LLMEngine(
            apiUrl=self.config.ollamaServer.url,
            timeout=self.config.ollamaServer.timeout,
            llmOptions=self.config.llmOptions,
            concurrencyPerModel=self.config.concurrencyPerModel,
            maxConcurrentModels=self.config.maxConcurrentModels,
            outputFile=str(self.paths.rawOutputPath),
            # 雖然 doBuildLLMTasks 已過濾，這裡再傳一次作為寫入端 dedup 安全網
            existingTaskIDs=completedIDSet,
        )

        logging.info(f"Dispatching {len(tasksToRunList)} tasks to async engine...")
        taskDictList = [task.model_dump() for task in tasksToRunList]
        asyncio.run(engineObj.doExecuteTaskBatches(taskDictList))
        logging.info(f"Inference complete. Raw output saved to: {self.paths.rawOutputPath}")

    # ===================== 主流程 =====================

    @staticmethod
    def _logStep(n: int, title: str):
        logging.info(f"==== [Step {n}] {title} ====")

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
        self._logStep(1, "Loading Task CSV & Prompts")
        taskDf = self.doLoadTaskCsv()
        promptCmbList = self.doLoadPromptCmb()
        self.doSavePromptPreview(taskDf, promptCmbList)

        self._logStep(2, "Building LLM Tasks")
        completedIDSet = self.doGetCompletedTasks()
        tasksToRunList = self.doBuildLLMTasks(taskDf, promptCmbList, completedIDSet)
        if not tasksToRunList and not completedIDSet:
            raise TaskBuildError("No tasks to run and no checkpoint found.")

        if tasksToRunList:
            self._logStep(3, f"Running Inference ({len(tasksToRunList)} tasks)")
            try:
                self.doRunInference(tasksToRunList, completedIDSet)
            except Exception as e:
                raise InferenceError(f"Inference failed: {e}") from e
        else:
            self._logStep(3, "All tasks completed. Skipping inference.")

        self._logStep(4, "Parsing LLM Outputs")
        parsedPath = OutputParser(
            rawCsvPath=self.paths.rawOutputPath,
            outputCsvPath=self.paths.resultPath,
            singlePromptCmbOutputDir=self.paths.singlePromptCmbOutputDir,
        ).run()

        self._logStep(5, "Processing Results")
        processedPath = LLMResultProcessor(
            inputCsvPath=parsedPath,
            outputCsvPath=self.paths.partialInfoPath,
            mergedPath=self.paths.fullInfoPath,
            labelMap=self.config.labelMap,
        ).run()

        self._logStep(6, "Evaluating")
        PromptCmbEval(
            inputCsvPath=processedPath,
            outputBaseDir=self.paths.evalDir,
        ).run()

        logging.info("Pipeline completed successfully!")
