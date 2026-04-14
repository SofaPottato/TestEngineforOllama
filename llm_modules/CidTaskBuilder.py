import logging
import pandas as pd
from typing import List, Dict, Any
from .schemas import LLMTask, TaskBuildError

class TaskBuilder:
    def __init__(
        self,
        models: List[str],
        pairNumber: int,
        pairBatchEnabled: bool,
        taskTemplate: str,
        itemTemplate: str = "Item {i}: Chemical: {e1} | Disease: {e2}\n",
        idCol: str = 'ID',
        pmidCol: str = 'PMID',
        titleCol: str = 'Title',
        abstractCol: str = 'Abstract',
        e1Col: str = 'E1_Name',
        e2Col: str = 'E2_Name',
        labelCol: str = 'Relation_Type',
        fallbackLabelCol: str = 'Label'
    ):
        """
        初始化任務建構器。
        負責將原始資料集與 Prompt 組合，排列組合成可送給 LLMEngine 執行的 LLMTask 清單。

        :param models: 要測試的模型名稱清單
        :param pairNumber: 每個任務批次包含的 pair 數量（pairBatchEnabled=False 時強制為 1）
        :param pairBatchEnabled: 是否啟用批次模式；False 時每個 pair 單獨送推論
        :param taskTemplate: 組裝 userPrompt 的文字模板，須含 {title}/{abstract}/{pairsContent} 佔位符
        :param itemTemplate: 單一 pair 的格式化模板，須含 {i}/{e1}/{e2} 佔位符
        :param idCol: 資料集中的資料 ID 欄位名稱
        :param pmidCol: PubMed ID 欄位名稱，用於以文章為單位分組
        :param titleCol: 文章標題欄位名稱
        :param abstractCol: 文章摘要欄位名稱
        :param e1Col: 第一個實體（Chemical）欄位名稱
        :param e2Col: 第二個實體（Disease）欄位名稱
        :param labelCol: 真實標籤的主要欄位名稱
        :param fallbackLabelCol: 主要欄位不存在時的備用標籤欄位名稱
        """
        self.modelList = models
        self.pairNumber = pairNumber if pairBatchEnabled else 1  # 未啟用批次模式時，每次只送一個 pair
        self.taskTemplate = taskTemplate
        self.itemTemplate = itemTemplate
        self.idCol = idCol
        self.pmidCol = pmidCol
        self.titleCol = titleCol
        self.abstractCol = abstractCol
        self.e1Col = e1Col
        self.e2Col = e2Col
        self.labelCol = labelCol
        self.fallbackLabelCol = fallbackLabelCol

        logging.info("TaskBuilder Initialized.")

    def doBuildLLMTasks(self, dataDf: pd.DataFrame, promptCmbList: List[Dict[str, str]], completedTasks: set = None) -> List[LLMTask]:
        """
        根據資料集與 Prompt 組合，建構所有待執行的 LLMTask 清單。

        建構流程：
        1. 依 PMID 將資料分組，同一篇文章的 pair 放在一起
        2. 每篇文章的 pair 清單依 pairNumber 切片，形成多個 batch
        3. 對每個模型 × prompt × batch 的組合建立一個 LLMTask
        4. 跳過 completedTasks 中已完成的任務（斷點續傳）

        :param dataDf: 原始資料集 DataFrame
        :param promptCmbList: Prompt 組合清單，每筆含 'promptID' 與 'promptText'
        :param completedTasks: 已完成任務的 taskID set，用於斷點續傳（None 時視為全部未完成）
        :return: 待執行的 LLMTask 清單
        :raises TaskBuildError: 缺少必要的 PMID 欄位時
        """
        logging.info("==== [TaskBuilder] Preparing Batched Tasks ====")
        tasksToRunList: List[LLMTask] = []
        if completedTasks is None:
            completedTasks = set()

        if self.pmidCol not in dataDf.columns:
            raise TaskBuildError(f"找不到分組欄位 '{self.pmidCol}'，請檢查原始資料！")

        if self.idCol not in dataDf.columns:
            logging.warning(f"⚠️ 找不到指定的 ID 欄位 '{self.idCol}'。")

        # 依 PMID 分組：確保同一篇文章的 pair 在同一個任務批次中，共享標題與摘要
        pmidGroupedDf = dataDf.groupby(self.pmidCol)
        articlePairBatchList = []

        for pmidID, pmidGroupDf in pmidGroupedDf:
            title = pmidGroupDf.iloc[0].get(self.titleCol, '')
            abstract = str(pmidGroupDf.iloc[0].get(self.abstractCol, ''))
            pairsList = []
            for idx, row in pmidGroupDf.iterrows():  # 將同一篇文章的所有 pair 整理成一個清單
                pairsList.append({
                    'orig_idx': idx,                                                         # DataFrame 原始索引，作為 Data_ID 傳至 OutputParser
                    'Data_ID': str(row.get(self.idCol, idx)),
                    'E1_Name': row.get(self.e1Col, ''),
                    'E2_Name': row.get(self.e2Col, ''),
                    'True_Label': row.get(self.labelCol, row.get(self.fallbackLabelCol, '')) # labelCol 不存在時 fallback
                })

            for i in range(0, len(pairsList), self.pairNumber):
                # 依 pairNumber 切片，每 N 個 pair 組成一個 batch
                batchPairsList = pairsList[i : i + self.pairNumber]
                articlePairBatchList.append({
                    'pmid': pmidID,
                    'title': title,
                    'abstract': abstract,
                    'batchPairsList': batchPairsList
                })

        skippedCount = 0

        for model in self.modelList:              # 所有模型
            for promptCmbDict in promptCmbList:   # 所有 prompt 組合
                promptID = promptCmbDict.get('promptID', promptCmbDict.get('id', 'Unknown'))
                sysPrompt = promptCmbDict.get('promptText', promptCmbDict.get('text', ''))

                for articlePairBatchDict in articlePairBatchList:   # 所有已分配好的批次
                    firstPairID = articlePairBatchDict['batchPairsList'][0]['Data_ID']
                    # taskID 格式：{model}::{promptID}::{firstPairID}，例如 gemma3:270m::EMO01::12345
                    currentTaskID = f"{model}::{promptID}::{firstPairID}"
                    if currentTaskID in completedTasks:
                        skippedCount += 1
                        continue

                    # 將每個 pair 依 itemTemplate 格式化，拼接成 pairsContent 區塊
                    pairsContent = ""
                    for i, pair in enumerate(articlePairBatchDict['batchPairsList'], 1):
                        pairsContent += self.itemTemplate.format(
                            i=i, e1=pair['E1_Name'], e2=pair['E2_Name']
                        )

                    # 用 str.replace 填入 taskTemplate 的佔位符，組裝完整的 userPrompt
                    userText = self.taskTemplate.replace('{title}', str(articlePairBatchDict['title']))
                    userText = userText.replace('{abstract}', str(articlePairBatchDict['abstract']))
                    userText = userText.replace('{pairsContent}', pairsContent)

                    taskObj = LLMTask(
                        taskID=currentTaskID,
                        model=model,
                        promptID=promptID,
                        sysPrompt=sysPrompt,
                        userPrompt=userText,
                        batchData=articlePairBatchDict  # 儲存整個 batch 的元資料，供 OutputParser 還原 pair 對應關係
                    )
                    tasksToRunList.append(taskObj)

        if skippedCount > 0:
            logging.info(f"已跳過 {skippedCount} 筆先前已完成的任務。")

        logging.info(f"成功建構 {len(tasksToRunList)} 筆需要執行的新任務批次！")
        return tasksToRunList
