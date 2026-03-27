import logging
import pandas as pd
from typing import List, Dict, Any
from .schemas import LLMTask, TaskBuildError # 確保有載入你寫的 Schema

class TaskBuilder:
    def __init__(
        self,
        models: List[str],
        pairNumber: int,
        taskTemplate: str,
        itemTemplate: str = "Item {i}: Chemical: {e1} | Disease: {e2}\n",
        idCol: str = 'ID',              # 🌟 新增：你的原生 ID 欄位名稱 (如果是其他名字請修改這裡)
        groupByCol: str = 'PMID',
        titleCol: str = 'Title',
        abstractCol: str = 'Abstract',
        e1Col: str = 'E1_Name',
        e2Col: str = 'E2_Name',
        labelCol: str = 'Relation_Type',
        fallbackLabelCol: str = 'Label'
    ):
        self.models = models
        self.pairNumber = pairNumber
        self.taskTemplate = taskTemplate
        self.itemTemplate = itemTemplate
        self.idCol = idCol              # 🌟 註冊 ID 欄位
        self.groupByCol = groupByCol
        self.titleCol = titleCol
        self.abstractCol = abstractCol
        self.e1Col = e1Col
        self.e2Col = e2Col
        self.labelCol = labelCol
        self.fallbackLabelCol = fallbackLabelCol
        
        logging.info("TaskBuilder Initialized.")

    def buildLLMInferenceTasks(self, df: pd.DataFrame, promptConfigsList: List[Dict[str, str]], completedTasks: set = None) -> List[LLMTask]:
        logging.info("==== [TaskBuilder] Preparing Batched Tasks ====")
        tasksList: List[LLMTask] = []
        if completedTasks is None:
            completedTasks = set()
            
        if self.groupByCol not in df.columns:
            raise TaskBuildError(f"找不到分組欄位 '{self.groupByCol}'，請檢查原始資料！")
            
        # 🌟 防呆：檢查有沒有你說的 ID 欄位
        if self.idCol not in df.columns:
            logging.warning(f"⚠️ 找不到指定的 ID 欄位 '{self.idCol}'，將暫時使用 DataFrame Index 代替。")

        grouped = df.groupby(self.groupByCol)
        baseBatchesList = []
        
        for groupId, group in grouped:
            title = group.iloc[0].get(self.titleCol, '')
            abstract = str(group.iloc[0].get(self.abstractCol, ''))
            
            pairsList = []
            for idx, row in group.iterrows():
                pairsList.append({
                    'orig_idx': idx,
                    'Data_ID': str(row.get(self.idCol, idx)), # 🌟 抓取原生 ID，若無則退回 index
                    'E1_Name': row.get(self.e1Col, ''),
                    'E2_Name': row.get(self.e2Col, ''),
                    'True_Label': row.get(self.labelCol, row.get(self.fallbackLabelCol, ''))
                })
            
            for i in range(0, len(pairsList), self.pairNumber):
                batchPairsList = pairsList[i : i + self.pairNumber]
                baseBatchesList.append({
                    'pmid': groupId, 
                    'title': title,
                    'abstract': abstract,
                    'batchPairsList': batchPairsList
                })
        
        skipped_count = 0 
        
        for model in self.models:
            for promptConfigDict in promptConfigsList:
                promptID = promptConfigDict.get('promptID', promptConfigDict.get('id', 'Unknown'))
                sysPrompt = promptConfigDict.get('promptText', promptConfigDict.get('text', ''))
                
                for batch in baseBatchesList:
                    # ==========================================
                    # 🌟 終極斷點邏輯：使用 [模型] + [PromptID] + [首筆原生ID]
                    # ==========================================
                    first_item_id = batch['batchPairsList'][0]['Data_ID']
                    current_task_id = f"{model}::{promptID}::{first_item_id}"
                    
                    if current_task_id in completedTasks:
                        skipped_count += 1
                        continue # 完美達成逐筆跳過！
                    
                    pairsContent = ""
                    for i, pair in enumerate(batch['batchPairsList'], 1):
                        pairsContent += self.itemTemplate.format(
                            i=i, e1=pair['E1_Name'], e2=pair['E2_Name']
                        )
                    
                    userText = self.taskTemplate.replace('{title}', str(batch['title']))
                    userText = userText.replace('{abstract}', str(batch['abstract']))
                    userText = userText.replace('{pairsContent}', pairsContent)
                    
                    task_obj = LLMTask(
                        task_id=current_task_id, # 🌟 寫入終極身分證
                        model=model,
                        promptID=promptID,
                        sysPrompt=sysPrompt,
                        userPrompt=userText,
                        batchData=batch
                    )
                    tasksList.append(task_obj)
                    
        if skipped_count > 0:
            logging.info(f"⏭️ 智慧過濾：已跳過 {skipped_count} 筆先前已完成的任務。")
            
        logging.info(f"✅ 成功建構 {len(tasksList)} 筆需要執行的新任務批次！")
        return tasksList