import yaml
from prompt_modules.PromptManager import PromptManager

def loadConfig(configPath="configs/prompt_config.yaml"):
    with open(configPath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    print("🚀 啟動階段一：生成與匯出 Prompt 組合")
    
    # 1. 讀取設定檔 
    cfgDict = loadConfig()
    
    # 2. 初始化管理器 
    pmObj = PromptManager(configDict=cfgDict)
    
    # 3. 執行排列組合邏輯 
    pmObj.generateCombinations()
    
    # 4. 呼叫匯出功能 
    csvPath = pmObj.exportPromptFiles()
    
    if csvPath:
        print("\n" + "="*50)
        print("Prompt 檔案已生成！")
        print(f"存放於{csvPath} ")
        print("="*50)