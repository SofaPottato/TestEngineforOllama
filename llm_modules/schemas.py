from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Dict, Any, List, Optional


class PathsConfig(BaseModel):
    dataPath: Path = Field(..., description="原始資料集 CSV 的路徑")
    promptCmbPath: Path = Field(..., description="Prompt 範本 CSV 的路徑")
    singlePromptCmbOutputDir: Path = Field(..., description="單一 Prompt 獨立輸出的目錄")
    promptCmbEvalDir: Path = Field(..., description="評估圖表與結果的輸出目錄")
    allPromptsCmbResultPath: Path = Field(..., description="解析後的 CSV 輸出路徑")
    promptCmbPartialInfoPath: Path = Field(..., description="清理後結果的 CSV 輸出路徑")
    promptCmbFullInfoPath: Path = Field(..., description="最終合併資料的 CSV 輸出路徑")
    rawPromptCmbOutputPath: Path = Field(..., description="暫存檔路徑")

    @field_validator('*')
    @classmethod
    def ensure_directories(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, Path) and ('Dir' in info.field_name):
            v.mkdir(parents=True, exist_ok=True)
        return v

class OllamaServerConfig(BaseModel):
    url: str = Field(default="http://localhost:11434/api/chat", description="Ollama API 端點")
    timeout: int = Field(default=1800, description="API 請求超時時間(秒)")

class PairConfig(BaseModel):
    pairNumbers: int = Field(default=10, description="每個 Prompt 要放入的資料對數量")
    enabled: bool = Field(default=True, description="是否啟用 Pair 批次模式；False 時每對單獨送推論")

class LLMAppConfig(BaseModel):
    """最外層的整體設定檔 Schema"""
    paths: PathsConfig
    selectedModels: List[str] = Field(default_factory=list, description="要進行測試的 LLM 模型清單")
    testLimits: Optional[int] = Field(default=None, description="開發測試用：限制讀取的資料筆數")
    ollamaServer: OllamaServerConfig = Field(default_factory=OllamaServerConfig)
    llmOptions: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0}, description="LLM 推論參數")
    pairSettings: PairConfig = Field(default_factory=PairConfig)
    concurrencyPerModel: int = Field(default=8, description="每個模型的最大非同步併發數")
    maxConcurrentModels: int = Field(default=1, description="最大同時運行的模型數量")
    taskTemplate: str = Field(
        default="{title}\n{abstract}\n{pairsContent}",
        description="組裝 Prompt 的預設文字模板"
    )

# ==========================================
# 1. 自定義異常 (Custom Exceptions)
# ==========================================
class PipelineError(Exception):
    """所有 Pipeline 錯誤的基底類別"""
    pass

class DataLoadError(PipelineError):
    """資料或 Prompt 載入失敗"""
    pass

class TaskBuildError(PipelineError):
    """任務建構失敗 (例如缺少必要欄位)"""
    pass

class InferenceError(PipelineError):
    """LLM 推論階段失敗"""
    pass

class ParsingError(PipelineError):
    """解析輸出失敗"""
    pass

# ==========================================
# 2. 資料結構定義 (Data Models)
# ==========================================
class LLMTask(BaseModel):
    """
    LLM 推論任務的標準資料結構。
    告別 Dict[str, Any]，讓每個欄位都有明確的家。
    """
    taskID: str = Field(..., min_length=1, description="任務唯一識別碼")
    model: str = Field(..., min_length=1, description="Ollama 模型名稱")
    promptID: str = Field(..., min_length=1, description="Prompt 策略識別碼")
    sysPrompt: str = Field(default="", description="系統提示詞")
    userPrompt: str = Field(..., min_length=1, description="使用者提示詞")
    batchData: Dict[str, Any] = Field(default_factory=dict, description="批次資料內容")