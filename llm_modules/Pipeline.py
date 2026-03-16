import logging
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

# 假設你把前面的設定管理員存在 utils.py 中
from llm_modules.utils import LLMConfigManager 
from llm_modules.LLM_Engine import ParallelInferenceEngine

class ExperimentPipeline:
    def __init__(self, config_manager: LLMConfigManager):
        # 1. 取得強型別的設定與 Path 物件
        self.cfg = config_manager.config
        self.paths: Dict[str, Path] = config_manager.paths
        
        # 2. 萃取給 Engine 的專屬設定 (不含路徑)
        api_config = self.cfg.get('ollama_server', {})
        llm_options = self.cfg.get('llm_hyperparameters', {})
        exec_settings = self.cfg.get('execution_settings', {})
        
        # 3. 初始化推論引擎 (傳入純設定，無路徑)
        self.engine = ParallelInferenceEngine(api_config, llm_options, exec_settings)

    def run(self):
        logging.info("📦 1. 載入資料與準備 Tasks...")
        # 這裡的 paths['data_path'] 已經是 pathlib.Path 物件，可以直接給 pandas 讀取！
        # df = pd.read_csv(self.paths['data_path'])
        
        # 建立任務清單 (範例假資料)
        tasks = self._build_tasks() 
        
        logging.info("🧠 2. 啟動 LLM 推論引擎...")
        # 引擎只負責運算，回傳結果 List
        completed_tasks = self.engine.run_tasks(tasks)
        
        logging.info("💾 3. 解析結果並儲存...")
        self._save_results(completed_tasks)

    def _build_tasks(self) -> List[Dict[str, Any]]:
        """建構給引擎的任務清單"""
        # 這裡實作你的 Prompt 組裝邏輯
        return [
            {
                "task_id": 1, 
                "model": "gemma3:270m", 
                "sys_prompt": "You are a bio expert...",
                "user_prompt": "Item 1: Chemical: Aspirin | Disease: Headache"
            }
        ]
        
    def _save_results(self, completed_tasks: List[Dict[str, Any]]):
        """將推論結果儲存為 CSV"""
        if not completed_tasks:
            logging.warning("⚠️ 沒有推論結果可供儲存。")
            return

        # 將 List 轉換為 DataFrame
        df = pd.DataFrame(completed_tasks)
        
        # 取得儲存路徑 (Path 物件)
        output_path: Path = self.paths['llm_merged_output_path']
        
        # 確保父目錄存在 (雖然 ConfigManager 已經建了，但雙重保險是好習慣)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 寫入 CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"✅ 最終結果已成功儲存至: {output_path}")