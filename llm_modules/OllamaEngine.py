import asyncio
import json
import logging
import time
import csv  
import os  
import httpx
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Union
from pathlib import Path  
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

class AsyncOllamaClient:
    """
    非同步的 API 客戶端，負責與 Ollama 伺服器進行通訊與錯誤重試。
    """
    def __init__(self, apiUrl: str, timeout: float, llmOptions: Dict[str, Any]):
        self.apiUrl = apiUrl
        self.timeout = timeout
        self.llmOptions = llmOptions

        limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
        self.client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout( self.timeout,connect=30.0)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, httpx.ReadTimeout)),
        reraise=True 
    )
    async def generate(self, modelName: str, sysPrompt: str, userPrompt: str):
        payload = {
            "model": modelName,
            "messages": [
                {"role": "system", "content": sysPrompt},
                {"role": "user", "content": userPrompt}
            ],
            "stream": False,
            "options": self.llmOptions
        }

        try:
            response = await self.client.post(self.apiUrl, json=payload)
            response.raise_for_status() 
            return response.json().get('message', {}).get('content', '')
        except Exception as e:
            logging.warning(f"⚠️ 模型 {modelName} 連線異常或逾時: {e}")
            raise 

    async def close(self):
        """關閉 HTTP 客戶端"""
        await self.client.aclose()


class MultiModelAsyncEngine:
    """
    支援多模型動態路由的非同步推論引擎。
    自動為不同的模型建立專屬的併發閘門。
    """
    def __init__(self,
                 apiUrl: str,
                 timeout: float,
                 llmOptions: Dict[str, Any],
                 concurrencyPerModel: int,
                 maxConcurrentModels: int,
                 outputFile: Union[str, Path],
                 existingTaskIds: set = None):

        self.concurrencyPerModel = concurrencyPerModel
        self.maxConcurrentModels = maxConcurrentModels
        self.outputFile = str(outputFile)

        self.client = AsyncOllamaClient(apiUrl=apiUrl, timeout=timeout, llmOptions=llmOptions)
        self.semaphores = defaultdict(lambda: asyncio.Semaphore(self.concurrencyPerModel))
        self.modelSemaphore = asyncio.Semaphore(self.maxConcurrentModels)
        self.existingTaskIds: set = existingTaskIds if existingTaskIds is not None else self._loadExistingTaskIds()
        self.debugPrintedModels = set()
        self.fileLock = asyncio.Lock()
    def _loadExistingTaskIds(self) -> set:
        if not os.path.isfile(self.outputFile):
            return set()
        try:
            df = pd.read_csv(self.outputFile, usecols=['task_id'], encoding='utf-8-sig')
            return set(df['task_id'].dropna().astype(str))
        except Exception:
            return set()
        
    async def processSingleTask(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """處理單一任務批次，並根據模型名稱進入對應的排隊閘門"""

        if isinstance(task, dict):#支援 dict 格式的任務資料結構，從字典中提取必要的欄位
            task_id = str(task.get('task_id', 'unknown_task_id'))
            model = str(task.get('model', 'unknown_model'))
            systemPrompt = task.get('sysPrompt', '')
            userPrompt = task.get('userPrompt', '')
            batchData = task.get('batchData', {})
        else:
            task_id = str(getattr(task, 'task_id', 'unknown_task_id'))
            model = str(getattr(task, 'model', 'unknown_model'))
            systemPrompt = getattr(task, 'sysPrompt', '')
            userPrompt = getattr(task, 'userPrompt', '')
            batchData = getattr(task, 'batchData', {})

        if task_id in self.existingTaskIds:
        #如果任務 ID 已存在於已完成的任務集合中，則直接返回原始任務資料，並在日誌中記錄跳過訊息
            logging.debug(f"任務 {task_id} 已存在於檔案，跳過")
            return task if isinstance(task, dict) else task.__dict__

        async with self.semaphores[model]:
            #根據模型名稱進入對應的併發閘門，確保同一時間只有指定數量的任務在使用該模型
            if model not in self.debugPrintedModels:
                logging.info(f"\nModel '{model}' 已啟動專屬排程，最大併發限制: {self.concurrencyPerModel}")
                self.debugPrintedModels.add(model)
            try:
                rawOutput = await self.client.generate(model, systemPrompt, userPrompt)
                #這裡是實際呼叫 Ollama API 的地方，並等待回應
            except Exception as e:
                logging.error(f"❌ 任務 {task_id} 失敗: {e}")
                rawOutput = "Error: Max retries exceeded or connection failed"

            if not rawOutput:
                rawOutput = "Error: Max retries exceeded or connection failed"

            completedTask = task.copy() if isinstance(task, dict) else task.__dict__.copy()
            completedTask['rawOutput'] = rawOutput
            completedTask['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

            batchDataStr = json.dumps(batchData, ensure_ascii=False)
            
            row_data = {
                "timestamp": completedTask['timestamp'],
                "task_id": task_id,
                "model": model,
                "promptID": completedTask.get('promptID', ''),
                "systemPrompt": systemPrompt,
                "userPrompt": userPrompt,
                "rawOutput": rawOutput,
                "batchData": batchDataStr 
            }
        
            async with self.fileLock:
                self.existingTaskIds.add(task_id)
                fileExists = os.path.isfile(self.outputFile)
                with open(self.outputFile, 'a', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row_data.keys())
                    if not fileExists:
                        writer.writeheader()
                    writer.writerow(row_data)

            return completedTask

    async def executeAsyncInferenceBatches(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """執行所有任務批次"""
        if not tasks:
            logging.warning("⚠️ 任務清單為空！")
            return []

        tasksByModel = defaultdict(list)#根據模型名稱將任務分組，確保同一模型的任務進入同一個併發閘門
        for task in tasks:
            tasksByModel[task.get('model')].append(task)

        uniqueModels = list(tasksByModel.keys())
        logging.info(f"開始推論！共 {len(tasks)} 筆任務批次。")
        logging.info(f"偵測到 {len(uniqueModels)} 種模型: {', '.join(uniqueModels)}")
        logging.info(f"允許同時運行的最大模型數量 (maxConcurrentModels): {self.maxConcurrentModels}")
        logging.info(f"每個模型最大併發限制 (concurrencyPerModel): {self.concurrencyPerModel}")

        pbar = tqdm(total=len(tasks), desc="總推論進度", unit="batch")

        async def processModelGroup(model_name: str, model_tasks: list):
            #為特定模型的任務批次建立專屬的併發閘門，確保同一時間只有指定數量的任務在使用該模型
            async with self.modelSemaphore:
                logging.info(f"模型 [{model_name}] 取得執行許可，開始處理 {len(model_tasks)} 筆任務...")
                
                coroutines = [self.processSingleTask(task) for task in model_tasks]
                
                results = []
                for f in asyncio.as_completed(coroutines):
                    result = await f
                    results.append(result)
                    pbar.update(1) 
                
                logging.info(f"模型 [{model_name}] 的任務已全數處理完畢，釋放模型許可。")
                return results

        model_group_coroutines = [ #為每個模型的任務批次創建一個協程，這些協程將根據模型名稱進入對應的排隊閘門
            processModelGroup(model_name, model_tasks)
            for model_name, model_tasks in tasksByModel.items()
        ]

        with logging_redirect_tqdm():
            grouped_results = await asyncio.gather(*model_group_coroutines)
            #等待所有模型的任務批次完成，並收集結果

        pbar.close()
        final_results = [item for sublist in grouped_results for item in sublist]
        return final_results