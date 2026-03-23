import pandas as pd
import os
import time
import re
import logging 

class LLMResultProcessor:
    def __init__(self, input_csv_path, output_dir,extra_merged_path,original_df ):
        """
        初始化資料處理器
        :param input_csv_path: 原始 LLM 輸出結果 (Raw Output)
        :param output_dir: 處理後檔案的儲存目錄
        """
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self.extra_merged_path = extra_merged_path
        self.original_df = original_df # 🌟 把它存起來
        self.required_cols = ['Model', 'Prompt_ID', 'Pred_Label', 'True_Label']
        self.index_cols = ['Data_ID', 'PMID', 'E1', 'E2']
        self.raw_df = None
        self.pivot_df = None
        
        logging.info(f"LLMResultProcessor(input_csv_path='{self.input_csv_path}', outputDir='{self.output_dir}')")

    def process(self):
        """
        [Public] 執行完整的處理流程：讀取 -> 解析 -> 轉置 -> 存檔
        :return: 處理後的檔案路徑 (str) 或 None
        """
        logging.info(f"process(self=<{self.__module__}.{self.__class__.__name__} object at {hex(id(self))}>)")
        
        logging.info(f"Processing data: {self.input_csv_path}")
        
        if not self._load_data():
            return None
 
        self.raw_df['Pred_Label'] = self.raw_df['Pred_Label'].apply(self._parse_response)
        
        logging.info("Processing True Labels")
        self.raw_df['True_Label'] = self.raw_df['True_Label'].apply(
            lambda x: 1 if str(x).strip().lower() == 'cid' else 0
        )

        logging.info("Creating Feature Names")
        self.raw_df['Feature_Name'] = self.raw_df['Model'].astype(str) + "_" + self.raw_df['Prompt_ID'].astype(str)
        

        if not self._pivot_data():
            return None
            
        result_path = self._save_data()  
        return result_path

    def _load_data(self):
        """[Private] 讀取並驗證 CSV"""
        logging.info("Loading Raw CSV Data...")
        if not os.path.exists(self.input_csv_path):
            logging.error(f"❌ Error: File not found: {self.input_csv_path}")
            return False
            
        try:
            self.raw_df = pd.read_csv(self.input_csv_path)
            logging.info(f"Data loaded successfully. Shape: {self.raw_df.shape}")
            
            missing = [c for c in self.required_cols + self.index_cols if c not in self.raw_df.columns]
            if missing:
                logging.error(f"❌ Error: Missing columns: {missing}")
                return False
            return True
            
        except Exception as e:
            logging.error(f"❌ Error reading CSV: {e}")
            return False

    def _parse_response(self, text):
        """
        [Private] 解析單一回應字串
        :return: 1 (Yes), 0 (No), -1 (Unknown)
        """
        if pd.isna(text):
            return -1
            
        s = str(text).lower().strip()
        
        s = s.replace('*', '').replace('`', '').replace('#', '').replace('"', '').replace("'", "")
        s = s.rstrip('.,!')

        if s in ['yes', '1']: return 1
        if s in ['no', '0']: return 0

        if re.search(r'\byes\b', s): return 1
        if re.search(r'\bno\b', s): return 0

        if 'positive' in s: return 1
        if 'negative' in s: return 0

        return -1 # 解析失敗

    def _pivot_data(self):
        """[Private] 將長表格轉為寬表格"""
        logging.info("Pivoting table (Long to Wide)")
        try:
            self.pivot_df = self.raw_df.pivot_table(
                index=self.index_cols + ['True_Label'], 
                columns='Feature_Name', 
                values='Pred_Label',
                aggfunc='first' 
            )
            
            # 整理表格
            self.pivot_df = self.pivot_df.reset_index()
            self.pivot_df = self.pivot_df.fillna(-1) 
            logging.info(f"Pivot completed. New Shape: {self.pivot_df.shape}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Pivot failed: {e}")
            return False

    def _save_data(self):
            """[Private] 儲存結果 (分流：一份簡單乾淨，一份豐富資訊)"""
            try:   
                save_path = os.path.join(str(self.output_dir), "processed_clean_results.csv")                # 1. 儲存給 Evaluate.py 用的「乾淨版本」
                self.pivot_df.to_csv(save_path, index=False, encoding='utf-8-sig')
                
                # ==========================================
                # 2. 獨立產生並儲存「人類閱讀用的豐富大表」
                # ==========================================
                if self.extra_merged_path and self.original_df is not None:
                    logging.info("Generating rich merged table for human review...")
                    
                    # 定義你想從原始資料抓過來的欄位
                    cols_to_add = ['Title', 'Abstract', 'Full_Text', 'E1_Type', 'E1_MeSH', 'E2_Type', 'E2_MeSH']
                    valid_cols = [c for c in cols_to_add if c in self.original_df.columns]
                    
                    # 複製需要的欄位，並把索引 (Index) 設為 Data_ID
                    orig_subset = self.original_df[valid_cols].copy()
                    orig_subset['Data_ID'] = orig_subset.index
                    
                    # 🌟 建立一個全新的 DataFrame 來合併，不污染原本的 self.pivot_df
                    rich_df = pd.merge(self.pivot_df, orig_subset, on='Data_ID', how='left')
                    
                    # 整理一下欄位順序
                    front_cols = ['Data_ID', 'PMID', 
                                'E1', 'E1_Type', 'E2', 'E2_Type',
                                'True_Label', 'Title', 'Abstract']
                    front_cols = [c for c in front_cols if c in rich_df.columns]
                    pred_cols = [c for c in rich_df.columns if c not in front_cols]
                    
                    rich_df = rich_df[front_cols + pred_cols]
                    
                    # 將豐富大表存到你額外指定的路徑
                    rich_df.to_csv(self.extra_merged_path, index=False, encoding='utf-8-sig')
                    logging.info(f"   - 🌟 [額外備份] 豐富資訊總成已儲存至: {self.extra_merged_path}")
                # ==========================================

                # 統計解析成功率
                valid_count = (self.raw_df['Pred_Label'] != -1).sum()
                total_count = len(self.raw_df)
                
                logging.info("✅ Data processed successfully!")
                logging.info(f"   - Clean Shape: {self.pivot_df.shape}")
                logging.info(f"   - Parse Success Rate: {valid_count}/{total_count} ({valid_count/total_count:.1%})")
                logging.info(f"   - Clean Pipeline Data Saved to: {save_path}")
                
                return save_path # 回傳的依然是給 Eval 用的乾淨路徑
                
            except Exception as e:
                logging.error(f"❌ Error saving file: {e}")
                return None