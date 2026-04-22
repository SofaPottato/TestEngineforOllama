# call_LLM.py (主程式)
import argparse
import logging
import sys
from llm_modules.utils import ReadLLMConfig, initializeGlobalLogger, setupSeed
from llm_modules.Pipeline import ExperimentPipeline

def startLLMPipeline() -> int:
    parser = argparse.ArgumentParser(description="LLM Inference Runner")
    parser.add_argument('--config', type=str, default='configs/PPI_config.yaml', help='Path to YAML configDict file')
    args = parser.parse_args()
    initializeGlobalLogger(logDir="./logs", logName="testLog.log")
    setupSeed(42)

    logging.info("========================================")
    logging.info("        Ollama            ")
    logging.info("========================================")

    try:
        configManager = ReadLLMConfig(args.config)
        pipeline = ExperimentPipeline(configManager.config)
        pipeline.run()
        return 0
    except Exception as e:
        logging.critical(f"發生未預期的錯誤: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(startLLMPipeline())