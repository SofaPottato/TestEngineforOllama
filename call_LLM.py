# call_LLM.py (主程式)
import argparse
import logging
from llm_modules.utils import ReadLLMConfig, initializeGlobalLogger, setupSeed
from llm_modules.Pipeline import ExperimentPipeline

def startLLMPipeline():
    parser = argparse.ArgumentParser(description="LLM Inference Runner")
    parser.add_argument('--config', type=str, default='configs/LLM_PPI_config.yaml', help='Path to YAML configDict file')
    args = parser.parse_args()
    initializeGlobalLogger(logDir="./logs", logName=f"testLog.log")
    setupSeed(42)
    
    logging.info("========================================")
    logging.info("        Ollama            ")
    logging.info("========================================")

    try:
        # 1. 載入並解析設定檔與路徑
        configManager = ReadLLMConfig(args.config)
        
        # 2. 將 ConfigManager 傳給 Pipeline
        pipeline = ExperimentPipeline(configManager.config)

        # 3. 執行
        pipeline.run()
        
    except Exception as e:
        logging.critical(f"❌ 發生未預期的錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    startLLMPipeline()