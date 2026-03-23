import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Dict, List, Any
import csv  
import os  
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm

class AsyncOllamaClient:
    """
    非同步的 API 客戶端，負責與 Ollama 伺服器進行通訊與錯誤重試。
    """
    def __init__(self, api_config: Dict[str, Any], llm_options: Dict[str, Any]):
        self.api_url = api_config.get('url', "http://localhost:11434/api/chat")
        self.timeout = api_config.get('timeout', 1800.0)
        self.llm_options = llm_options

        # 建立高效的非同步 HTTP 連線池
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
        self.client = httpx.AsyncClient(limits=limits, timeout=self.timeout)

    # Tenacity 重試裝飾器：最多試 3 次，等待時間為 1, 2, 4 秒
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, httpx.ReadTimeout)),
        reraise=False # 重試 3 次都失敗後，回傳 None 交給外層處理
    )
    async def generate(self, model_name: str, sys_prompt: str, user_prompt: str):
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": self.llm_options
        }

        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status() 
            return response.json().get('message', {}).get('content', '')
        except Exception as e:
            logging.warning(f"⚠️ 模型 {model_name} 連線異常或逾時: {e}")
            raise # 拋給 tenacity 進行重試

    async def close(self):
        """關閉 HTTP 客戶端"""
        await self.client.aclose()


class MultiModelAsyncEngine:
    """
    支援多模型動態路由的非同步推論引擎。
    會自動為不同的模型建立專屬的併發閘門，確保不互相搶佔資源。
    """
    def __init__(self, api_config: Dict[str, Any], llm_options: Dict[str, Any], exec_settings: Dict[str, Any]):
        self.concurrencyPerModel = exec_settings.get('concurrencyPerModel', 8)
        self.outputFile = exec_settings.get('outputFile', 'results.csv')
        
        self.client = AsyncOllamaClient(api_config, llm_options)
        
        # 🌟 核心魔法：自動為出現的「每個新模型」配發一個獨立的 Semaphore (併發閘門)
        self.semaphores = defaultdict(lambda: asyncio.Semaphore(self.concurrencyPerModel))
        
        self.debugPrintedModels = set()
        self.fileLock = asyncio.Lock()
        
    async def processSingleTask(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """處理單一任務批次，並根據模型名稱進入對應的排隊閘門"""
        model = task.get('model', 'unknown_model')
        
        async with self.semaphores[model]:
            systemPrompt = task.get('sys_prompt', '')
            userPrompt = task.get('user_prompt', '')

            # 每個模型只印出一次啟動提示
            if model not in self.debugPrintedModels:
                logging.info(f"\n📢 [Debug] Model '{model}' 已啟動專屬排程，最大併發限制: {self.concurrencyPerModel}")
                self.debugPrintedModels.add(model)

            # 呼叫 API
            rawOutput = await self.client.generate(model, systemPrompt, userPrompt)
            
            # 如果重試 3 次都失敗
            if rawOutput is None:
                rawOutput = "Error: Max retries exceeded or connection failed"

            completedTask = task.copy()
            completedTask['rawOutput'] = rawOutput
            completedTask['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

            # ==========================================
            # 🌟 改為即時寫入 CSV 的邏輯
            # ==========================================
            # 把複雜的 batch_data 轉成字串，以免破壞 CSV 格式
            batchDataStr = json.dumps(completedTask.get('batch_data', {}), ensure_ascii=False)
            
            # 定義 CSV 的欄位與要寫入的資料
            row_data = {
                "timestamp": completedTask['timestamp'],
                "model": completedTask.get('model', ''),
                "prompt_id": completedTask.get('prompt_id', ''),
                "systemPrompt": completedTask.get('sys_prompt', ''),
                "userPrompt": completedTask.get('user_prompt', ''),
                "rawOutput": rawOutput,
                "batchData": batchDataStr # 複雜結構以字串形式存入單一儲存格
            }

            # 檢查檔案是否已存在 (用來決定要不要寫入標題列 Header)
            fileExists = os.path.isfile(self.outputFile)

            # 使用 csv.DictWriter 安全地寫入 (自動處理內文的逗號與換行)
            # newline='' 是為了防止 Windows 底下 CSV 產生多餘的空行
            async with self.fileLock:
                fileExists = os.path.isfile(self.outputFile)
                with open(self.outputFile, 'a', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row_data.keys())
                    if not fileExists:
                        writer.writeheader()
                    writer.writerow(row_data)

            return completedTask

    async def runTasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """執行所有任務批次"""
        if not tasks:
            logging.warning("⚠️ 任務清單為空！")
            return []

        uniqueModels = set(t.get('model') for t in tasks)
        logging.info(f"🚀 開始非同步推論！共 {len(tasks)} 筆任務批次。")
        logging.info(f"⚙️ 偵測到 {len(uniqueModels)} 種模型: {', '.join(uniqueModels)}")
        logging.info(f"⚙️ 每個模型最大併發限制: {self.concurrencyPerModel}")

        # 建立所有協程
        coroutines = [self.processSingleTask(task) for task in tasks]
        
        # 併發執行並顯示進度條
        results = await tqdm.gather(*coroutines, desc="總推論進度", unit="batch")
        
        await self.client.close()
        logging.info(f"✅ 所有任務執行完畢！原始 JSONL 備份已儲存至 {self.outputFile}")
        return results