from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, List, Optional, ClassVar


class PathsConfig(BaseModel):
    """
    路徑設定。
    - 必填：taskCsvPath、promptCmbPath、outputRoot
    - 其餘輸出路徑未填 → 依約定自動衍生到 outputRoot 底下
    - 填相對路徑 → 視為相對 outputRoot
    - 填絕對路徑 → 原樣使用，不受 outputRoot 影響
    """
    # --- 輸入 ---
    taskCsvPath:   Path = Field(..., description="前處理產出的標準 Task CSV 路徑")
    promptCmbPath: Path = Field(..., description="Prompt 組合 CSV 的路徑")

    # --- 輸出根目錄（所有未明確指定的輸出路徑都會放在這底下） ---
    outputRoot:    Path = Field(..., description="輸出檔案的根目錄")

    # --- 輸出路徑（皆可 override；未填時套用 _DEFAULT_NAMES） ---
    rawOutputPath:         Optional[Path] = Field(default=None, description="LLM 推論原始暫存 CSV 路徑")
    resultPath:            Optional[Path] = Field(default=None, description="OutputParser 解析後的結構化 CSV 路徑")
    singlePromptCmbOutputDir: Optional[Path] = Field(default=None, description="每個 promptID 獨立存檔的目錄")
    partialInfoPath:       Optional[Path] = Field(default=None, description="Pivot 後的寬格式 CSV 路徑")
    fullInfoPath:          Optional[Path] = Field(default=None, description="合併原始欄位的完整版 CSV 路徑")
    evalDir:               Optional[Path] = Field(default=None, description="評估圖表與報表的輸出目錄")
    promptPreviewPath:     Optional[Path] = Field(default=None, description="渲染後的 userPrompt 預覽 CSV 路徑")

    # 未填時使用的預設檔名（相對 outputRoot）
    _DEFAULT_NAMES: ClassVar[Dict[str, str]] = {
        'rawOutputPath':         'raw.csv',
        'resultPath':            'result.csv',
        'singlePromptCmbOutputDir': 'singleOutput',
        'partialInfoPath':       'partialInfo.csv',
        'fullInfoPath':          'fullInfo.csv',
        'evalDir':               'eval',
        'promptPreviewPath':     'prompt_preview.csv',
    }

    @model_validator(mode='after')
    def resolveAndEnsureDirectories(self):
        """
        1. 先依 outputRoot 解析所有輸出路徑：
             None            → outputRoot / 預設檔名
             相對路徑         → outputRoot / 該相對路徑
             絕對路徑         → 原樣
        2. 再統一 mkdir：
             欄位名含 'Dir'   → 目錄本身
             其餘（檔案路徑） → parent 目錄
        """
        for name, default in self._DEFAULT_NAMES.items():
            current: Optional[Path] = getattr(self, name)
            if current is None:
                resolved = self.outputRoot / default
            elif not current.is_absolute():
                resolved = self.outputRoot / current
            else:
                resolved = current
            setattr(self, name, resolved)

        # 集中 mkdir
        for fieldName in self.model_fields:
            value = getattr(self, fieldName)
            if isinstance(value, Path):
                target = value if 'Dir' in fieldName or fieldName == 'outputRoot' else value.parent
                target.mkdir(parents=True, exist_ok=True)

        return self


class OllamaServerConfig(BaseModel):
    url: str = Field(default="http://localhost:11434/api/chat", description="Ollama API 端點")
    timeout: int = Field(default=1800, description="API 請求超時時間(秒)")


class LabelMapConfig(BaseModel):
    """標籤映射設定：將原始標籤統一轉成 1 (正類) / 0 (負類)"""
    positive: List[str] = Field(default_factory=lambda: ["1", "true", "yes"])
    negative: List[str] = Field(default_factory=lambda: ["0", "false", "no", "none", "negative"])


class LLMAppConfig(BaseModel):
    """
    簡化後的 Pipeline 設定檔 Schema。
    資料集專屬邏輯已移至前處理腳本，Pipeline 只需要：
    - 路徑、模型、LLM 參數
    - taskTemplate / pairTemplate（使用者自由編輯 prompt 格式）
    - labelMap（通用的 label 轉換）
    """
    paths: PathsConfig
    selectedModels: List[str] = Field(default_factory=list, description="要進行測試的 LLM 模型清單")
    contextColumns: List[str] = Field(default_factory=list, description="Task CSV 中作為 context 的欄位名稱（對應 taskTemplate 的佔位符，如 title、abstract）")
    pairColumns: List[str] = Field(default_factory=list, description="pairs JSON 中對應 pairTemplate 佔位符的欄位名稱（如 e1、e2）")
    ollamaServer: OllamaServerConfig = Field(default_factory=OllamaServerConfig)
    llmOptions: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0}, description="LLM 推論參數")
    labelMap: LabelMapConfig = Field(default_factory=LabelMapConfig)
    pairNumber: int = Field(default=1, description="每個 LLM task 包含的 item 數；1 = 單筆模式，>1 = 批次模式")
    concurrencyPerModel: int = Field(default=8, description="每個模型的最大非同步併發數")
    maxConcurrentModels: int = Field(default=1, description="最大同時運行的模型數量")
    taskTemplate: str = Field(
        ...,
        description="組裝 userPrompt 的文字模板；{key} 對應 Task CSV 的 context 欄位，{pairs} 對應批次展開"
    )
    pairTemplate: Optional[str] = Field(
        default=None,
        description="單筆 pair 的格式化模板（如 '{i}: {e1} | {e2}'）；不設定代表單筆模式"
    )


# ==========================================
# 自定義異常 (Custom Exceptions)
# ==========================================
class PipelineError(Exception):
    """所有 Pipeline 錯誤的基底類別"""
    pass

class DataLoadError(PipelineError):
    """資料或 Prompt 載入失敗"""
    pass

class TaskBuildError(PipelineError):
    """任務建構失敗"""
    pass

class InferenceError(PipelineError):
    """LLM 推論階段失敗"""
    pass

class ParsingError(PipelineError):
    """解析輸出失敗"""
    pass


# ==========================================
# 資料結構定義 (Data Models)
# ==========================================
class LLMTask(BaseModel):
    """LLM 推論任務的標準資料結構"""
    taskID: str = Field(..., min_length=1, description="任務唯一識別碼")
    model: str = Field(..., min_length=1, description="Ollama 模型名稱")
    promptID: str = Field(..., min_length=1, description="Prompt 策略識別碼")
    sysPrompt: str = Field(default="", description="系統提示詞")
    userPrompt: str = Field(..., min_length=1, description="使用者提示詞")
    pairs: List[Dict[str, Any]] = Field(default_factory=list, description="此任務包含的 pair 清單（含 id/label）")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task 層級 context 欄位（title/abstract/passage 等，由 contextColumns 決定）")
