import yaml
import logging
import os
import random
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any

class LLMConfigManager:
    """專門負責讀取 LLM 設定檔並處理型別與路徑"""
    
    def __init__(self, config_path: str):
        # 使用 Dict[str, Any] 確保型別清晰
        self.config: Dict[str, Any] = self.loadYaml(config_path)
        
        # 將 paths 區塊獨立抽出來，轉換為 Path 物件
        self.paths: Dict[str, Path] = self.parsePaths()
        
    def loadYaml(self, path: str) -> Dict[str, Any]:
        """讀取 YAML 檔案"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def parsePaths(self) -> Dict[str, Path]:
        """將 YAML 中的 paths 字串轉換為 pathlib.Path 物件"""
        parsedPaths: Dict[str, Path] = {}
        rawPaths: Dict[str, str] = self.config.get('paths', {})
        
        for key, pathStr in rawPaths.items():
            pathObj = Path(pathStr)
            parsedPaths[key] = pathObj
            
            # 如果是輸出資料夾 (結尾是 _dir)，自動幫忙建立目錄
            if key.endswith('_dir'):
                pathObj.mkdir(parents=True, exist_ok=True)
                
        return parsedPaths

    def getModels(self) -> List[str]:
        """取得模型清單 (回傳 List[str])"""
        return self.config.get('selectedModels', [])
    
def loadConfigs(path):
    """
    讀取 YAML 設定檔
    :param path: 設定檔路徑 (例如 'config/llm_config.yaml')
    :return: 字典格式的設定內容
    """
    if not os.path.exists(path):
        logging.critical(f"❌ Critical Error: Config file not found at {path}")
        raise RuntimeError("Data loading failed")
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logging.info(f"✅ 設定檔已載入: {path}")
            return config
    except Exception as e:
        logging.error(f"❌ Error loading config: {e}")
        raise RuntimeError("Data loading failed")

def setupLogger(logDir="./logs", logName="experiment.log"):
    """
    設定全域 Logger，同時輸出到檔案與終端機
    :param logDir: Log 檔案存放目錄
    :param logName: Log 檔名
    """
    os.makedirs(logDir, exist_ok=True)
    logPath = os.path.join(logDir, logName)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s", 
        force=True, 
        handlers=[
            logging.FileHandler(logPath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)               
        ]
    )
    logging.info(f"📝 Logger initialized. Writing to {logPath}")

def setupSeed(seed=42):
    """
    固定隨機種子，確保實驗可重現 (Reproducibility)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed}")