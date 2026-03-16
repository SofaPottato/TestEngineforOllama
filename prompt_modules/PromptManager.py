import yaml
import os
import logging
import pandas as pd
from itertools import combinations, product
from pathlib import Path

# ==========================================
# 負責處理資料夾與設定讀取
# ==========================================
class PromptConfig:
    @staticmethod
    def loadYaml(yamlPath):
        if not os.path.exists(yamlPath):
            logging.error(f"❌ 找不到檔案: {yamlPath}")
            raise FileNotFoundError(f"❌ 找不到檔案: {yamlPath}")
        
        with open(yamlPath, 'r', encoding='utf-8') as f:
            dataDict = yaml.safe_load(f)
        return dataDict.get('prompts', dataDict)

    @staticmethod
    def ensureDirectories(outputDirPath):
        outputDirPath = Path(outputDirPath)
        outputDirPath.mkdir(parents=True, exist_ok=True)
        print(f"📁 已確認輸出目錄：\n   - {outputDirPath}\n")
        return outputDirPath

# ==========================================
# 負責核心的 Prompt 生成邏輯 
# ==========================================
class PromptGenerator:
    def __init__(self, methodPoolDict, configDict):
        self.methodPoolDict = methodPoolDict
        self.cfgDict = configDict
        self.generatedPromptList = []

    def generate(self):
        promptMode = self.cfgDict.get('promptMode', 'auto').lower()
        logging.info(f"🔧 Prompt Generation Mode: {promptMode.upper()}")

        if promptMode == 'auto':
            self.generateAutoMode()
        elif promptMode == 'manual':
            self.generateManualMode()
        else:
            logging.error(f"❌ 錯誤: 未知的模式 '{promptMode}'")
            return []

        self.sortResults()
        logging.info(f"✅ 成功生成 {len(self.generatedPromptList)} 組 Prompt。")
        return self.generatedPromptList

    def sortResults(self):
        def sortKey(itemDict):
            partsList = itemDict['id'].split(' + ')
            return (len(partsList), partsList)
        try:
            self.generatedPromptList.sort(key=sortKey)
        except Exception as e:
            logging.warning(f"⚠️ 排序時發生錯誤，跳過排序: {e}")

    def generateAutoMode(self):
        settingsDict = self.cfgDict.get('autoSettingsDict', {})
        targetMethodList = settingsDict.get('methods', list(self.methodPoolDict.keys()))
        maxSize = settingsDict.get('max_size', len(targetMethodList))
        limitNum = min(maxSize, len(targetMethodList))
        
        for r in range(1, limitNum + 1):
            for methodComboTuple in combinations(targetMethodList, r):
                promptList = []
                for cat in methodComboTuple:
                    if cat in self.methodPoolDict:
                        promptItemsList = [
                            (f"{cat}{str(k).zfill(2)}", v) 
                            for k, v in self.methodPoolDict[cat].items()
                        ]
                        promptList.append(promptItemsList)
                
                for itemComboTuple in product(*promptList):
                    idList = [item[0] for item in itemComboTuple]
                    textList = [item[1] for item in itemComboTuple]
                    self.addCombination(idList, textList)

    def generateManualMode(self):
            manualKeyList = self.cfgDict.get('manualKeysList', [])
            flatPoolDict = {}
        
            for catStr, itemsDict in self.methodPoolDict.items():
                if not isinstance(itemsDict, dict): 
                    continue
                for k, textStr in itemsDict.items():
                    kStr = str(k).zfill(2) 
                    fullIdStr = f"{catStr}{kStr}" 
                    flatPoolDict[fullIdStr] = {'id': fullIdStr, 'text': textStr.strip()}

            for comboKeyList in manualKeyList:
                idList, textList = [], []
                for itemKeyStr in comboKeyList:
                    if itemKeyStr in flatPoolDict:
                        idList.append(flatPoolDict[itemKeyStr]['id'])
                        textList.append(flatPoolDict[itemKeyStr]['text'])
                    else:
                        print(f"⚠️ 警告: 找不到指定的 Prompt ID '{itemKeyStr}'，將跳過此項目。")
                
                if idList:

                    self.addCombination(idList, textList)
                    
    def addCombination(self, idList, textList):
        self.generatedPromptList.append({
            "id": " + ".join(idList),
            "text": "\n".join(textList)
        })

# ==========================================
# 負責資料匯出 (隨時可抽換成匯出 JSON/Excel)
# ==========================================
class PromptExporter:
    def __init__(self, outputDirPath, fileName):
        self.outputDirPath = Path(outputDirPath)
        self.fileName = fileName

    def exportToCsv(self, generatedPromptList):
        if not generatedPromptList:
            print("⚠️ 警告：目前沒有任何生成的 Prompt 可以匯出！")
            return None

        csvDataList = [{"Prompt_ID": p['id'], "Prompt_Text": p['text']} for p in generatedPromptList]
        promptDf = pd.DataFrame(csvDataList)
        
        csvPath = self.outputDirPath / f"{self.fileName}.csv"
        promptDf.to_csv(csvPath, index=False, encoding='utf-8-sig')
        
        print(f"📊 CSV 檔案已儲存至: {csvPath}")
        return csvPath

# ==========================================
# 指揮官 (Manager) - 整合以上模組
# ==========================================
class PromptManager:
    def __init__(self, configDict):
        self.cfgDict = configDict
        
        # 1. 處理設定與路徑
        self.yamlPath = configDict.get("promptYamlPath", "configs/prompts.yaml")
        rawOutputDirPath = configDict.get('promptListOutputPath', 'prompt_output')
        self.outputDirPath = PromptConfig.ensureDirectories(rawOutputDirPath)
        self.fileName = configDict.get("promptListFileName", "generated_prompt_list")
        
        # 2. 讀取資料
        self.methodPoolDict = PromptConfig.loadYaml(self.yamlPath)
        
        # 3. 初始化生成器
        self.generatorObj = PromptGenerator(self.methodPoolDict, self.cfgDict)
        self.generatedPromptList = []

    def generateCombinations(self):
        """呼叫生成器進行邏輯運算"""
        self.generatedPromptList = self.generatorObj.generate()
        return self.generatedPromptList

    def exportPromptFiles(self):
        """呼叫匯出器處理存檔"""
        print(f"\n🔄 正在準備輸出 Prompt 列表至目錄: {self.outputDirPath} ...")
        exporterObj = PromptExporter(self.outputDirPath, self.fileName)
        return exporterObj.exportToCsv(self.generatedPromptList)