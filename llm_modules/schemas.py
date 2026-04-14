from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Dict, Any, List, Optional


class PathsConfig(BaseModel):
    taskCsvPath: Path = Field(..., description="前處理產出的標準 Task CSV 路徑")
    promptCmbPath: Path = Field(..., description="Prompt 組合 CSV 的路徑")
    rawOutputPath: Path = Field(..., description="LLM 推論原始暫存 CSV 路徑")
    resultPath: Path = Field(..., description="OutputParser 解析後的結構化 CSV 路徑")
    singlePromptOutputDir: Path = Field(..., description="每個 promptID 獨立存檔的目錄")
    partialInfoPath: Path = Field(..., description="Pivot 後的寬格式 CSV 路徑")
    fullInfoPath: Path = Field(..., description="合併原始欄位的完整版 CSV 路徑")
    evalDir: Path = Field(..., description="評估圖表與報表的輸出目錄")

    @field_validator('*')
    @classmethod
    def ensureDirectories(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, Path) and ('Dir' in info.field_name):
            v.mkdir(parents=True, exist_ok=True)
        return v


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
    - taskTemplate / itemTemplate（使用者自由編輯 prompt 格式）
    - labelMap（通用的 label 轉換）
    """
    paths: PathsConfig
    selectedModels: List[str] = Field(default_factory=list, description="要進行測試的 LLM 模型清單")
    ollamaServer: OllamaServerConfig = Field(default_factory=OllamaServerConfig)
    llmOptions: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0}, description="LLM 推論參數")
    labelMap: LabelMapConfig = Field(default_factory=LabelMapConfig)
    concurrencyPerModel: int = Field(default=8, description="每個模型的最大非同步併發數")
    maxConcurrentModels: int = Field(default=1, description="最大同時運行的模型數量")
    taskTemplate: str = Field(
        ...,
        description="組裝 userPrompt 的文字模板；{key} 對應 Task CSV 的 context 欄位，{items} 對應批次展開"
    )
    itemTemplate: Optional[str] = Field(
        default=None,
        description="單筆 item 的格式化模板（如 '{i}: {e1} | {e2}'）；不設定代表單筆模式"
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
    items: List[Dict[str, Any]] = Field(default_factory=list, description="此任務包含的 item 清單（含 id/label）")
