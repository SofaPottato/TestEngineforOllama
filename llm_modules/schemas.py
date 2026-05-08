"""Pydantic schema、自定義例外、資料模型的單一事實來源。"""
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, List, Optional, ClassVar, FrozenSet


# 任何處理 pair 的模組都應引用此常數，避免 'itemID'/'label' 硬編碼散落多處
RESERVED_PAIR_FIELDS: FrozenSet[str] = frozenset({'itemID', 'label'})


# ── Config schema（依組合關係排序：葉子 → 容器）───────────────────────────────
class PathsConfig(BaseModel):
    """
    路徑設定。輸出路徑未填時依 _DEFAULT_NAMES 衍生到 outputRoot；
    相對路徑視為相對 outputRoot；絕對路徑原樣使用。
    """
    taskCsvPath:   Path = Field(..., description="前處理產出的標準 Task CSV 路徑")
    promptCmbPath: Path = Field(..., description="Prompt 組合 CSV 的路徑")
    outputRoot:    Path = Field(..., description="輸出檔案的根目錄")

    rawOutputPath:            Optional[Path] = Field(default=None, description="LLM 推論原始暫存 CSV 路徑")
    resultPath:               Optional[Path] = Field(default=None, description="OutputParser 解析後的結構化 CSV 路徑")
    singlePromptCmbOutputDir: Optional[Path] = Field(default=None, description="每個 promptID 獨立存檔的目錄")
    partialInfoPath:          Optional[Path] = Field(default=None, description="Pivot 後的寬格式 CSV 路徑")
    fullInfoPath:             Optional[Path] = Field(default=None, description="合併原始欄位的完整版 CSV 路徑")
    evalDir:                  Optional[Path] = Field(default=None, description="評估圖表與報表的輸出目錄")
    promptPreviewPath:        Optional[Path] = Field(default=None, description="渲染後的 userPrompt 預覽 CSV 路徑")

    _DEFAULT_NAMES: ClassVar[Dict[str, str]] = {
        'rawOutputPath':            'raw.csv',
        'resultPath':               'result.csv',
        'singlePromptCmbOutputDir': 'singleOutput',
        'partialInfoPath':          'partialInfo.csv',
        'fullInfoPath':             'fullInfo.csv',
        'evalDir':                  'eval',
        'promptPreviewPath':        'prompt_preview.csv',
    }

    @model_validator(mode='after')
    def resolveAndEnsureDirectories(self):
        """依 outputRoot 解析所有輸出路徑（None/相對/絕對三種情形），並統一 mkdir。"""
        for name, default in self._DEFAULT_NAMES.items():
            current: Optional[Path] = getattr(self, name)
            if current is None:
                resolved = self.outputRoot / default
            elif not current.is_absolute():
                resolved = self.outputRoot / current
            else:
                resolved = current
            setattr(self, name, resolved)

        for fieldName in self.model_fields:
            value = getattr(self, fieldName)
            if isinstance(value, Path):
                target = value if 'Dir' in fieldName or fieldName == 'outputRoot' else value.parent
                target.mkdir(parents=True, exist_ok=True)

        return self


class OllamaServerConfig(BaseModel):
    """Ollama 伺服器連線設定。預設指向本機 11434 port 的 chat 端點。"""
    url: str = Field(default="http://localhost:11434/api/chat", description="Ollama API 端點")
    timeout: int = Field(default=1800, description="API 請求超時時間(秒)")


class LabelMapConfig(BaseModel):
    """
    標籤映射設定。positive / negative 供精確比對（== 語意，可含 "1"/"true"）；
    outputPositive / outputNegative 供 substring 掃描（in 語意，不可含 "1"/"true" 以免誤觸）。
    未命中一律標 -1。
    """
    positive: List[str] = Field(default_factory=lambda: ["1", "true", "yes"])
    negative: List[str] = Field(default_factory=lambda: ["0", "false", "no", "none", "negative"])
    outputPositive: List[str] = Field(
        default_factory=lambda: ["yes", "positive"],
        description="OutputParser 對 LLM 輸出做 substring 掃描的正類關鍵字"
    )
    outputNegative: List[str] = Field(
        default_factory=lambda: ["no", "negative", "none"],
        description="OutputParser 對 LLM 輸出做 substring 掃描的負類關鍵字"
    )


class LLMAppConfig(BaseModel):
    """
    Pipeline 設定 Schema。
    single-target（pairColumns 為空）：每個 task 一個預測標的，需設 labelColumn。
    multi-target（pairColumns 非空）：每個 task 多個預測標的，需設 pairTemplate。
    """
    paths: PathsConfig
    selectedModels: List[str] = Field(default_factory=list, description="要進行測試的 LLM 模型清單")
    contextColumns: List[str] = Field(default_factory=list, description="Task CSV 中作為 context 的欄位名稱")
    pairColumns: List[str] = Field(default_factory=list, description="pairs JSON 中對應 pairTemplate 佔位符的欄位名稱；為空代表 single-target 模式")
    labelColumn: Optional[str] = Field(default=None, description="single-target 模式下攜帶 true label 的欄位名")
    ollamaServer: OllamaServerConfig = Field(default_factory=OllamaServerConfig)
    llmOptions: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0}, description="LLM 推論參數")
    labelMap: LabelMapConfig = Field(default_factory=LabelMapConfig)
    maxPairsPerBatch: int = Field(default=1, description="每個 LLM task 包含的 item 數；>1 為批次模式")
    concurrencyPerModel: int = Field(default=8, description="每個模型的最大非同步併發數")
    maxConcurrentModels: int = Field(default=1, description="最大同時運行的模型數量")
    taskTemplate: str = Field(..., description="組裝 userPrompt 的文字模板；{key} 對應 context 欄位，{pairs} 對應批次展開")
    pairTemplate: Optional[str] = Field(default=None, description="單筆 pair 的格式化模板；不設定代表單筆模式")

    @property
    def isSingleTarget(self) -> bool:
        """pairColumns 為空 → single-target；否則 multi-target。"""
        return not self.pairColumns

    @model_validator(mode='after')
    def validateTargetMode(self):
        """single / multi-target 一致性檢查：必填欄位、禁用欄位、maxPairsPerBatch 限制。"""
        if self.isSingleTarget:
            if not self.labelColumn:
                raise ValueError(
                    "single-target 模式（pairColumns 為空）必須設定 labelColumn，"
                    "用於指定 Task CSV 中攜帶 true label 的欄位名。"
                )
            if self.pairTemplate is not None:
                raise ValueError(
                    "single-target 模式（pairColumns 為空）不應設定 pairTemplate。"
                    "若資料集為一篇多 pair，請設定 pairColumns。"
                )
            if self.maxPairsPerBatch != 1:
                raise ValueError(
                    f"single-target 模式下 maxPairsPerBatch 必須為 1，目前為 {self.maxPairsPerBatch}。"
                )
        else:
            if not self.pairTemplate:
                raise ValueError(
                    "multi-target 模式（pairColumns 非空）必須提供 pairTemplate，"
                    "否則無法將 pair 渲染進 userPrompt。"
                )
        return self


# ── 任務執行單位 ──────────────────────────────────────────────────────────
class LLMTask(BaseModel):
    """
    一次 LLM API 呼叫的輸入結構。
    唯一性由 composite key (model, promptID, taskID) 保證，不用字串拼接以避免特殊字元歧義。
    """
    taskID: str = Field(..., min_length=1, description="批次層級識別碼")
    model: str = Field(..., min_length=1, description="Ollama 模型名稱")
    promptID: str = Field(..., min_length=1, description="Prompt 策略識別碼")
    sysPrompt: str = Field(default="", description="系統提示詞")
    userPrompt: str = Field(..., min_length=1, description="使用者提示詞")
    pairs: List[Dict[str, Any]] = Field(default_factory=list, description="此任務包含的 pair 清單（含 itemID/label）")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task 層級 context 欄位")


# ── Pipeline 例外體系 ─────────────────────────────────────────────────────
class PipelineError(Exception):
    """所有 Pipeline 錯誤的基底類別，call_LLM.py 統一捕捉。"""
    pass

class DataLoadError(PipelineError):
    """資料或 Prompt 載入失敗：檔案不存在、欄位缺漏、格式錯誤等。"""
    pass

class TaskBuildError(PipelineError):
    """任務建構失敗：模型/prompt 清單為空、JSON 欄位無法解析等。"""
    pass

class InferenceError(PipelineError):
    """LLM 推論階段失敗。"""
    pass

class ParsingError(PipelineError):
    """解析輸出失敗：找不到 raw.csv、解析後無有效資料等。"""
    pass
