import logging
import os
import sys
from pathlib import Path
import pandas as pd

from .LLMEngine import InferenceManager
from .LLMResultProcessor import LLMResultProcessor
from .Evaluate import LLMEvaluationSystem

class ExperimentPipeline:

    def __init__(self, config):
        """
        初始化實驗流程
        :param config: 從 yaml 讀入的完整設定字典
        """
        logging.info("Initializing ExperimentPipeline()")
        self.cfg = config
        pathsConfig = config.get("paths", {})
        # 從 pathsConfig 中提取所有需要的路徑
        self.dataPath = Path(pathsConfig.get("dataPath", "data/bcvcdr_raw/BCVCDR_Processed.csv"))
        self.resultOutputPath = Path(pathsConfig.get("resultOutputPath", "data/llm_output/"))
        self.evalDataDir = Path(pathsConfig.get("evalDataDir", "data/llm_output/"))
        self.evalDataDir.mkdir(parents=True, exist_ok=True)
        self.promptsPath = Path(pathsConfig.get("promptsPath", "data/prompt_output/generatedPromptList.csv"))
        self.mergedOutputPath = Path(pathsConfig.get("mergedLlmOutputPath", "data/llm_output/merged_result.csv"))
        self.mergedOutputPath.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"ExperimentPipeline initialized, resultOutputPath='{self.resultOutputPath}'")

    def run(self):
        """執行實驗流程"""

        logging.info(f"==== [Step 1] Loading Data from: {self.dataPath} ====")
        df = self.loadData()
        if df is None:
            logging.critical("❌ Data loading failed. Pipeline aborted.")
            raise RuntimeError("Data loading failed")

        # ---------------------------------------------------------
        # 修改區塊：[Step 2] 改為直接從 CSV 讀取 Prompts
        # ---------------------------------------------------------
        logging.info(f"==== [Step 2] Loading Prompts from: {self.promptsPath} ====")
        prompts = self.loadPrompts(self.promptsPath)

        if not prompts:
            logging.error("❌ No prompts loaded. Aborting.")
            return

        logging.info("==== [Step 3] Running LLM ====")
        engine = InferenceManager(self.cfg)
        # 這裡傳入的 prompts 已經是 [{'Prompt_ID': '...', 'Prompt_Text': '...'}, ...] 的格式
        rawCsvPath = engine.run(df, prompts)

        if not rawCsvPath:
            logging.error("❌ Inference failed or produced no output.")
            return

        logging.info("==== [Step 4] Processing Results ====")
        processedDir = self.resultOutputPath
        processor = LLMResultProcessor(
            input_csv_path=rawCsvPath,
            output_dir=str(processedDir),
            extra_merged_path=str(self.mergedOutputPath),
            original_df=df  # <-- 把原始資料傳進去！
        )
        processedCsvPath = processor.process()

        if not processedCsvPath:
            logging.error("❌ Data processing failed.")
            return

        logging.info("==== [Step 5] Evaluate ====")
        eval_dir = self.evalDataDir
        evaluator = LLMEvaluationSystem(processedCsvPath, str(eval_dir))

        evaluator.runEvaluation()
        evaluator.analyzeDifficulty()
        evaluator.plotConfusionMatrices()
        evaluator.plotHeatmap()
        evaluator.saveResults()

    def loadData(self):
        """讀取原始資料"""
        if not self.dataPath or not os.path.exists(self.dataPath):
            logging.error(f"❌ Data file not found: {self.dataPath}")
            return None

        try:
            df = pd.read_csv(self.dataPath)

            if self.cfg.get('testLimits'):
                limit = self.cfg['testLimits']
                df = df.head(limit)
                logging.warning(f"⚠️test:Using only first {limit} pairs.")

            logging.info(f"✅ Data loaded. Shape: {df.shape}")
            return df

        except Exception as e:
            logging.error(f"❌ Failed to load data: {e}")
            return None

    # ---------------------------------------------------------
    # 新增區塊：專屬的 Prompt 讀取模組
    # ---------------------------------------------------------
    def loadPrompts(self, path: Path) -> list:
        """
        讀取 Prompt CSV 檔案並轉換為字典列表。
        這樣做可以確保 Pipeline 與特定的 Prompt 生成邏輯解耦。
        """
        if not path.exists():
            logging.error(f"❌ Prompt CSV file not found: {path}")
            return []

        try:
            # 讀取您提供的 test_manual_prompts.csv
            promptsDf = pd.read_csv(path)

            # 檢查是否包含預期的欄位，避免後續 LLM Engine 抓不到資料
            if 'Prompt_ID' not in promptsDf.columns or 'Prompt_Text' not in promptsDf.columns:
                logging.error("❌ CSV missing required columns: 'Prompt_ID' or 'Prompt_Text'")
                return []

            # 將 DataFrame 轉換為 list of dictionaries
            # 格式範例：[{'Prompt_ID': 'EMO01 + RAR02', 'Prompt_Text': 'This is very important...'}, ...]
            promptsList = promptsDf.to_dict(orient='records')
            logging.info(f"✅ Successfully loaded {len(promptsList)} prompts.")

            return promptsList

        except Exception as e:
            logging.error(f"❌ Failed to load prompt CSV: {e}")
            return []