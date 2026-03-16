import time
import requests
import logging
import threading 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from typing import Dict, List, Any

class OllamaClient:
    """
    純粹的 API 客戶端，負責與 Ollama 伺服器進行通訊與錯誤重試。
    """
    def __init__(self, api_config: Dict[str, Any], llm_options: Dict[str, Any]):
        self.api_url = api_config.get('url', "http://localhost:11434/api/chat")
        self.timeout = api_config.get('timeout', 1800)
        self.max_retries = api_config.get('max_retries', 3)
        self.llm_options = llm_options
        
        # 設定 HTTP 連線池，提升大量併發時的效能
        self.session = requests.Session() 
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def generate(self, model_name: str, sys_prompt: str, user_prompt: str) -> str:
        """發送請求並回傳 LLM 文字結果"""
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": self.llm_options
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(self.api_url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json().get('message', {}).get('content', '')
                else:
                    logging.warning(f"⚠️ API Error (Attempt {attempt+1}): HTTP {response.status_code} - {response.text}")
            except Exception as e:
                logging.warning(f"⚠️ Connection Error (Attempt {attempt+1}): {e}")
            
            # 發生錯誤時，等待一段時間再重試 (1秒, 2秒, 4秒...)
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        logging.error(f"❌ Error: Model {model_name} max retries exceeded.")
        return "Error: Max retries exceeded"


class ParallelInferenceEngine:
    """
    通用的多執行緒推論引擎。
    只接收網路與執行緒設定，不參與任何檔案 I/O 操作。
    """
    def __init__(self, api_config: Dict[str, Any], llm_options: Dict[str, Any], exec_settings: Dict[str, Any]):
        self.is_parallel = exec_settings.get('parallel', False)
        self.max_workers = exec_settings.get('max_workers', 3)
        self.model_concurrent_requests = exec_settings.get('model_concurrent_requests', 1)
        
        # 實例化 API 客戶端
        self.client = OllamaClient(api_config, llm_options)
        
        self._debug_lock = threading.Lock()
        self._debug_printed = False

    def run_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        執行推論任務。
        :param tasks: List of Dict. 必須包含 'model', 'sys_prompt', 'user_prompt' 等鍵值。
        :return: List of Dict. 回傳原本的任務字典，並新增 'raw_output' 鍵值儲存 LLM 回應。
        """
        if not tasks:
            logging.warning("⚠️ 接收到的任務清單為空！")
            return []

        logging.info(f"🚀 開始執行推論，共 {len(tasks)} 筆任務...")
        logging.info(f"⚙️ 執行模式: {'平行處理 (Parallel)' if self.is_parallel else '序列處理 (Sequential)'}")
        logging.info(f"⚙️ 最大執行緒數: {self.max_workers}")

        results = []
        # 如果不啟用平行處理，則強制為 1 個 Worker
        workers = self.max_workers if self.is_parallel else 1

        # 使用 tqdm 顯示進度條
        with tqdm(total=len(tasks), desc="推論進度", unit="task") as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # 提交所有任務到執行緒池
                future_to_task = {
                    executor.submit(self._process_single_task, task): task 
                    for task in tasks
                }
                
                # 收集完成的任務結果
                for future in as_completed(future_to_task):
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as e:
                        logging.error(f"❌ 任務執行發生嚴重錯誤: {e}")
                    finally:
                        pbar.update(1)
        
        logging.info("✅ 所有推論任務執行完畢！")
        return results

    def _process_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """[Private] 處理單一原子任務，呼叫 Client 並回傳結果"""
        model = task.get('model', 'unknown_model')
        sys_p = task.get('sys_prompt', '')
        user_p = task.get('user_prompt', '')
        
        # 僅印出一次 Debug 資訊確認格式正確，避免控制台被洗版
        if not self._debug_printed:
            with self._debug_lock:
                if not self._debug_printed:
                    logging.info(f"\n📢 [Debug] Model: {model} \n📢 [Debug] System: {sys_p[:100]}...\n📢 [Debug] User: {user_p[:100]}...")
                    self._debug_printed = True
            
        # 呼叫 Ollama API 進行推論
        raw_output = self.client.generate(model, sys_p, user_p)
        
        # 複製原本的任務資料，並附加上 LLM 的輸出結果與時間戳記
        completed_task = task.copy()
        completed_task['raw_output'] = raw_output
        completed_task['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return completed_task