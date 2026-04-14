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

class OllamaClient:
    """
    非同步的 API 客戶端，負責與 Ollama 伺服器進行通訊與錯誤重試。
    """
    def __init__(self, apiUrl: str, timeout: float, llmOptions: Dict[str, Any]):
        """
        :param apiUrl: Ollama API 端點（通常是 http://localhost:11434/api/chat）
        :param timeout: 單次請求的超時時間（秒）
        :param llmOptions: 傳給 Ollama 的推論參數（如 temperature、top_p 等）
        """
        self.apiUrl = apiUrl
        self.timeout = timeout
        self.llmOptions = llmOptions

        # 設定連線池上限，避免大量併發時建立過多 TCP 連線導致資源耗盡
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
        self.httpClientObj = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(self.timeout, connect=30.0)  # 整體請求 timeout 與連線建立 timeout 分開設定
        )

    @retry(
        stop=stop_after_attempt(3),                                         # 最多重試 3 次
        wait=wait_exponential(multiplier=1, min=1, max=10),                 # 指數退避：1s → 2s → 4s（上限 10s）
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, httpx.ReadTimeout)),
        reraise=True                                                         # 超過重試上限後，將例外往上拋給呼叫端
    )
    async def doGenerate(self, modelName: str, sysPrompt: str, userPrompt: str):
        """
        向 Ollama 發送單次推論請求並回傳模型的文字回應。
        發生網路錯誤或逾時時會自動重試（最多 3 次，指數退避）。

        :param modelName: Ollama 模型名稱（如 "llama3.2:1b"）
        :param sysPrompt: 系統提示詞
        :param userPrompt: 使用者提示詞（包含文章內容與待判斷的 pair）
        :return: 模型回應的文字字串
        :raises httpx.RequestError / httpx.ReadTimeout: 超過重試上限後往上拋出
        """
        payloadDict = {
            "model": modelName,
            "messages": [
                {"role": "system", "content": sysPrompt},
                {"role": "user", "content": userPrompt}
            ],
            "stream": False,          # 關閉串流，等待完整回應後一次回傳
            "options": self.llmOptions
        }

        try:
            responseObj = await self.httpClientObj.post(self.apiUrl, json=payloadDict)
            responseObj.raise_for_status()  # 4xx/5xx 狀態碼會拋出 HTTPStatusError，觸發重試
            return responseObj.json().get('message', {}).get('content', '')
        except Exception as e:
            logging.warning(f"⚠️ 模型 {modelName} 連線異常或逾時: {e}")
            raise

    async def doClose(self):
        """關閉 HTTP 客戶端，釋放底層 TCP 連線池"""
        await self.httpClientObj.aclose()


class LLMEngine:
    """
    支援多模型動態路由的非同步推論引擎。
    自動為不同的模型建立專屬的併發閘門（Semaphore），
    並透過 maxConcurrentModels 控制同時運行的模型數量，避免 GPU/VRAM 競爭。
    """
    def __init__(self,
                 apiUrl: str,
                 timeout: float,
                 llmOptions: Dict[str, Any],
                 concurrencyPerModel: int,
                 maxConcurrentModels: int,
                 outputFile: Union[str, Path],
                 existingTaskIds: set = None):
        """
        :param apiUrl: Ollama API 端點
        :param timeout: 單次請求的超時時間（秒）
        :param llmOptions: 傳給 Ollama 的推論參數
        :param concurrencyPerModel: 每個模型最大同時進行的推論請求數
        :param maxConcurrentModels: 最大同時運行的模型數（超過時後續模型需排隊等待）
        :param outputFile: 推論結果的 CSV 暫存檔路徑（斷點續傳的核心）
        :param existingTaskIds: 已完成任務的 taskID set（由外部傳入，避免重複讀檔）
        """
        self.concurrencyPerModel = concurrencyPerModel
        self.maxConcurrentModels = maxConcurrentModels
        self.outputFile = str(outputFile)
        self.ollamaClientObj = OllamaClient(apiUrl=apiUrl, timeout=timeout, llmOptions=llmOptions)
        # defaultdict 讓每個模型第一次存取時自動建立專屬 Semaphore，無需手動初始化
        self.modelSemaphoreDict = defaultdict(lambda: asyncio.Semaphore(self.concurrencyPerModel))
        self.modelConcurrencySemaphore = asyncio.Semaphore(self.maxConcurrentModels)  # 跨模型的全域閘門
        # 優先使用外部傳入的 set，避免重複讀取 CSV；若未傳入則從暫存檔自行載入
        self.existingTaskIDSet: set = existingTaskIds if existingTaskIds is not None else self._doLoadExistingTaskIDs()
        self.loggedModelSet = set()    # 記錄已印過啟動訊息的模型，避免重複 log
        self.fileLockObj = asyncio.Lock()  # 保護 CSV 的並發寫入，確保多個 coroutine 不會同時寫檔

    def _doLoadExistingTaskIDs(self) -> set:
        """
        從暫存 CSV 讀取已完成的 taskID set，用於 existingTaskIds 未由外部傳入時的 fallback。

        :return: 已完成任務的 taskID set；檔案不存在或讀取失敗時回傳空 set
        """
        if not os.path.isfile(self.outputFile):
            return set()
        try:
            checkpointDf = pd.read_csv(self.outputFile, usecols=['taskID'], encoding='utf-8-sig')
            return set(checkpointDf['taskID'].dropna().astype(str))
        except Exception:
            return set()

    async def doProcessSingleTask(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        處理單一推論任務批次：
        1. 檢查 taskID 是否已在 existingTaskIDSet（斷點續傳）
        2. 進入模型專屬 Semaphore 排隊（concurrencyPerModel 限流）
        3. 呼叫 Ollama API 取得回應
        4. 取得 fileLock 後寫入 CSV 暫存檔

        :param task: LLMTask 序列化後的 dict，必要欄位：taskID/model/sysPrompt/userPrompt/batchData
        :return: 包含 rawOutput 與 timestamp 的完整任務 dict
        """
        if isinstance(task, dict):  # 支援 dict 格式的任務資料結構，從字典中提取必要的欄位
            taskID = str(task.get('taskID', 'unknown_taskID'))
            model = str(task.get('model', 'unknown_model'))
            systemPrompt = task.get('sysPrompt', '')
            userPrompt = task.get('userPrompt', '')
            items = task.get('items', [])
        else:
            taskID = str(getattr(task, 'taskID', 'unknown_taskID'))
            model = str(getattr(task, 'model', 'unknown_model'))
            systemPrompt = getattr(task, 'sysPrompt', '')
            userPrompt = getattr(task, 'userPrompt', '')
            items = getattr(task, 'items', [])

        if taskID in self.existingTaskIDSet:
            # 若 taskID 已存在於已完成集合中，直接返回原始任務資料，不重送 API 請求
            logging.debug(f"任務 {taskID} 已存在於檔案，跳過")
            return task if isinstance(task, dict) else task.__dict__

        async with self.modelSemaphoreDict[model]:
            # 進入模型專屬 Semaphore：同一模型的併發請求數不超過 concurrencyPerModel
            if model not in self.loggedModelSet:
                logging.info(f"\nModel '{model}' 已啟動專屬排程，最大併發限制: {self.concurrencyPerModel}")
                self.loggedModelSet.add(model)
            try:
                rawOutput = await self.ollamaClientObj.doGenerate(model, systemPrompt, userPrompt)
                # 這裡是實際呼叫 Ollama API 的地方，並等待回應
            except Exception as e:
                logging.error(f"❌ 任務 {taskID} 失敗: {e}")
                rawOutput = "Error: Max retries exceeded or connection failed"

            if not rawOutput:
                rawOutput = "Error: Max retries exceeded or connection failed"

            completedTaskDict = task.copy() if isinstance(task, dict) else task.__dict__.copy()
            completedTaskDict['rawOutput'] = rawOutput
            completedTaskDict['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

            itemsJsonStr = json.dumps(items, ensure_ascii=False)  # 序列化為 JSON 字串存入 CSV

            rowDataDict = {
                "timestamp": completedTaskDict['timestamp'],
                "taskID": taskID,
                "model": model,
                "promptID": completedTaskDict.get('promptID', ''),
                "systemPrompt": systemPrompt,
                "userPrompt": userPrompt,
                "rawOutput": rawOutput,
                "items": itemsJsonStr
            }

            async with self.fileLockObj:
                # fileLock 保護：確保同一時間只有一個 coroutine 在寫 CSV，避免資料交錯
                self.existingTaskIDSet.add(taskID)  # 在鎖內更新，確保不會有重複寫入的競態條件
                b_fileExists = os.path.isfile(self.outputFile)
                with open(self.outputFile, 'a', encoding='utf-8-sig', newline='') as f:
                    csvWriterObj = csv.DictWriter(f, fieldnames=rowDataDict.keys())
                    if not b_fileExists:
                        csvWriterObj.writeheader()  # 首次建檔才寫表頭
                    csvWriterObj.writerow(rowDataDict)

        return completedTaskDict

    async def doExecuteTaskBatches(self, taskDictList: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        執行所有推論任務批次。
        將任務依模型分組，每個模型的任務群組各自取得 modelConcurrencySemaphore 後並發執行。
        各群組內部使用 asyncio.as_completed 讓先完成的任務優先更新進度條。

        :param taskDictList: 所有待執行的 LLMTask dict 清單
        :return: 所有任務的完成結果 List（含 rawOutput）
        """
        if not taskDictList:
            logging.warning("⚠️ 任務清單為空！")
            return []

        # 根據模型名稱將任務分組，確保同一模型的任務進入同一個併發閘門
        tasksByModelDict = defaultdict(list)
        for task in taskDictList:
            tasksByModelDict[task.get('model')].append(task)

        uniqueModelList = list(tasksByModelDict.keys())
        logging.info(f"開始推論！共 {len(taskDictList)} 筆任務批次。")
        logging.info(f"偵測到 {len(uniqueModelList)} 種模型: {', '.join(uniqueModelList)}")
        logging.info(f"允許同時運行的最大模型數量 (maxConcurrentModels): {self.maxConcurrentModels}")
        logging.info(f"每個模型最大併發限制 (concurrencyPerModel): {self.concurrencyPerModel}")

        progressBarObj = tqdm(total=len(taskDictList), desc="總推論進度", unit="batch")

        async def doProcessModelGroup(modelName: str, modelTaskList: list):
            # 取得全域模型 Semaphore：超過 maxConcurrentModels 的模型需在此排隊
            async with self.modelConcurrencySemaphore:
                logging.info(f"模型 [{modelName}] 取得執行許可，開始處理 {len(modelTaskList)} 筆任務...")

                coroutineList = [self.doProcessSingleTask(task) for task in modelTaskList]

                resultList = []
                # as_completed 讓先完成的 coroutine 優先回傳，即時更新 tqdm 進度條
                for f in asyncio.as_completed(coroutineList):
                    result = await f
                    resultList.append(result)
                    progressBarObj.update(1)

                logging.info(f"模型 [{modelName}] 的任務已全數處理完畢，釋放模型許可。")
                return resultList

        # 為每個模型建立對應的 group coroutine，透過 gather 同時啟動所有模型群組
        modelGroupCoroutineList = [
            doProcessModelGroup(modelName, modelTaskList)
            for modelName, modelTaskList in tasksByModelDict.items()
        ]

        with logging_redirect_tqdm():
            # logging_redirect_tqdm 確保 logging 輸出不會干擾 tqdm 進度條的顯示
            groupedResultList = await asyncio.gather(*modelGroupCoroutineList)

        progressBarObj.close()
        # 攤平二維 List（每個模型群組各產生一個 list）為一維
        finalResultList = [item for sublist in groupedResultList for item in sublist]
        return finalResultList
