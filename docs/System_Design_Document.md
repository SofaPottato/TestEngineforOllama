# 專案架構與系統設計說明書
## LLM-Based Relation Extraction Evaluation System

**版本：** 1.0　｜　**作者：** Chen BOBO　｜　**日期：** 2026-04-07

---

## 一、系統定位與業務目標

本系統是一套針對**生物醫學文本關係抽取（Chemical-Induced Disease Relation Extraction）**的 LLM 評估框架。其核心任務是：對多個本地部署的 LLM（透過 Ollama 提供服務），以多種 Prompt 策略進行組合測試，自動化量測各組合在 CID（Chemical-Induced Disease）分類任務上的精準度，並產出可供分析的統計報表與視覺化圖表。

**這不是一個推論服務，而是一個實驗評估引擎。** 其設計目標是：快速迭代 Prompt 策略、橫向比較模型表現、以及透過可重現的批量實驗取得可靠的量化結論。

---

## 二、整體架構概覽

```
┌──────────────────────────────────────────────────────────────┐
│                       call_LLM.py                            │
│              (Entry Point / CLI Bootstrapper)                │
└─────────────────────────┬────────────────────────────────────┘
                          │ argparse → config path
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  utils.ReadLLMConfig                         │
│         YAML → Pydantic LLMAppConfig (型別驗證層)            │
└─────────────────────────┬────────────────────────────────────┘
                          │ LLMAppConfig (validated)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              Pipeline.ExperimentPipeline                     │
│                  (Orchestration Layer)                       │
│                                                              │
│  Step 1 ──► doLoadDataSet / doLoadPromptCmb                  │
│  Step 2 ──► TaskBuilder.doBuildLLMTasks       ◄── 斷點續傳   │
│  Step 3 ──► LLMEngine.doExecuteTaskBatches    ◄── 非同步引擎 │
│  Step 4 ──► OutputParser.doParse              ◄── Regex 解析 │
│  Step 5 ──► LLMResultProcessor.doCleanAndMerge               │
│  Step 6 ──► PromptCmbEval.doEval / doSaveResults             │
└──────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    CSV / Pivot      Confusion        eval_summary
    Wide Table       Matrices PNG       .csv
```

整個系統採用**六階段線性 Pipeline 架構**，每個階段職責單一、邊界清晰，階段間以檔案路徑（`Path`）作為契約傳遞，而非記憶體物件——這是一個有意識的設計選擇，後面會進一步說明其合理性。

---

## 三、組件職責分析

### 3.1 配置層：`schemas.py` + `utils.ReadLLMConfig`

這是整個系統的**配置契約層**，採用 Pydantic `BaseModel` 作為 schema 驗證引擎。

```
YAML File → yaml.safe_load() → dict → LLMAppConfig(**dict)
                                              ↓
                              PathsConfig (field_validator: 自動建立 Dir)
                              OllamaServerConfig
                              PairConfig
                              LLMAppConfig (頂層聚合)
```

**設計優點：**

- `PathsConfig.ensure_directories` 透過 `field_validator('*')` 在 schema 驗證階段自動建立所有帶 `Dir` 後綴的目錄，確保後續流程永遠不會因「目錄不存在」而在預期外的地方失敗。程式的真正執行邏輯完全不需要處理目錄建立的責任。
- `rawPromptCmbOutputPath` 宣告為 `Field(...)` 必填項，而非 `Optional`。這讓配置錯誤在 `ReadLLMConfig.__init__` 的 Pydantic 解析階段就立即失敗，而非在 Pipeline Step 3 執行推論時才崩潰並丟出難以定位的 `AttributeError`。這是**「讓錯誤早死」（Fail Fast）**原則的具體實踐。

---

### 3.2 協調層：`Pipeline.ExperimentPipeline`

Pipeline 是整個系統的**指揮中心**，不包含任何業務邏輯，只負責協調各組件的執行順序與資料傳遞。

值得注意的是其**斷點續傳（Checkpoint-Resume）機制**：

```python
# Pipeline.doGetCompletedTasks()
completedIDSet = checkpointDf['taskID'].dropna().astype(str).str.strip().tolist()

# Pipeline.run()
tasksToRunList = taskBuilderObj.doBuildLLMTasks(..., completedTasks=completedIDSet)

if tasksToRunList:
    # 有新任務才執行推論
else:
    # 全部完成，直接跳到解析階段
```

**機制設計：** `taskID` 的格式為 `{model}::{promptID}::{firstPairData_ID}`，三個維度的組合唯一標識一個任務。即使中途停止，再次執行時只有尚未完成的任務會被重新推論，`LLMEngine` 以 append 模式逐筆寫入暫存 CSV，不需要 transaction 或 rollback 機制。

這個設計對於長時間實驗（數小時）的容錯性至關重要，避免因網路中斷或主機重啟而導致需要從頭重跑。

---

### 3.3 任務建構層：`TaskBuilder`

`TaskBuilder.doBuildLLMTasks` 的核心邏輯是一個**三層巢狀迭代**：

```
for model in modelList:              # M 個模型
  for prompt in promptCmbList:       # P 個 Prompt
    for batch in articlePairBatchList:  # B 個批次（以 PMID 為單位切分）
      → 產生一個 LLMTask
```

**總任務量 = M × P × B**

以 PMID（PubMed 論文 ID）作為分組單位，而非直接以單個 pair 為單位，這有兩個作用：

1. **上下文完整性**：同一篇文章的所有 pair 共享同一份 Title/Abstract，放入同一個批次讓 LLM 可以在相同的上下文中回答多個問題，符合真實閱讀情境。
2. **效率**：相比每個 pair 單獨發送一次 API 請求，批次模式（`pairSettings.pairNumbers: 10`）可將 API 呼叫次數降低約 10 倍。

`taskTemplate` 與 `itemTemplate` 以字串佔位符（`{title}`, `{e1}` 等）注入，讓 Prompt 的結構完全由 YAML 配置控制，**無需修改程式碼即可調整實驗的 Prompt 格式**。

---

### 3.4 非同步推論引擎：`OllamaEngine`

這是整個系統中架構複雜度最高的組件，採用**雙層 Semaphore 限流設計**：

```
Level 1: modelConcurrencySemaphore（全域）
  └── 控制同時運行的模型數量 (maxConcurrentModels=1)
      → 防止多個大型模型同時佔用 VRAM 導致 OOM

Level 2: modelSemaphoreDict[model]（per-model）
  └── 控制單一模型的最大並發請求數 (concurrencyPerModel=8)
      → 防止對單一模型過度併發導致 Ollama 排隊崩潰
```

**實現方式：**

```python
# defaultdict 讓每個模型第一次存取時自動建立專屬 Semaphore
self.modelSemaphoreDict = defaultdict(lambda: asyncio.Semaphore(self.concurrencyPerModel))

async def doProcessModelGroup(modelName, modelTaskList):
    async with self.modelConcurrencySemaphore:          # Level 1
        for f in asyncio.as_completed(coroutineList):   # Level 2 在 doProcessSingleTask 內部
            result = await f
```

**`asyncio.as_completed` vs `asyncio.gather` 的選擇：** 在 `doProcessModelGroup` 內部使用 `as_completed`，讓先完成的任務優先更新 `tqdm` 進度條，而非等待整批完成後才更新。對長時間執行的推論，這對使用者體驗有顯著差異。

**重試機制：** `OllamaClient.doGenerate` 使用 `tenacity` 裝飾器，對 `httpx.RequestError / HTTPStatusError / ReadTimeout` 進行最多 3 次、指數退避（1s → 2s → 4s）的自動重試，並設 `reraise=True`。這確保瞬斷可以自動恢復，但持續性故障不會被靜默吞掉，仍會向上傳遞。

**並發寫入保護：** `asyncio.Lock()` 保護 CSV 的並發寫入，確保多個 coroutine 完成時不會發生資料交錯。

---

### 3.5 輸出解析層：`OutputParser`

這是系統中**最脆弱的環節**，其穩健性完全依賴 LLM 輸出格式的一致性。

`doExtractAnswers` 的解析策略分三步：

```python
# Step 1. 清除粗體格式（**Yes** → Yes）
text = text.replace('*', '')

# Step 2. 在前方補換行，讓第一題也能被 split 捕捉
text = "\n" + text.strip()

# Step 3. 正則切割（核心）：匹配各種編號格式
blocksList = re.split(r'\n\s*(?:\s+|No\.?\s*)?\d+\s*[:.)-]', text, flags=re.IGNORECASE)
```

**設計的謹慎之處：** Pattern 強制要求以 `\n` 開頭，這是為了避免將答案文字中的 "No." 誤解析為題號分隔符（如 "No. 2 correlation found" 中的 "No."）。

解析失敗的結果標記為 `-1`（而非 `0`），確保「無法判斷」不會被誤算為「負類（No）」，影響評估指標的準確性。

---

### 3.6 資料處理層：`LLMResultProcessor`

**Long-to-Wide Pivot** 是本層的核心操作：

```
InputDf (長表格):
Model         promptID   Data_ID   Pred_Label
llama3.2_1b   EMO01      0         1
llama3.2_1b   EMO01      1         0
llama3.2_1b   RAR01      0         1
...

↓ pivot_table(index=[Data_ID,...], columns='Feature_Name', values='Pred_Label')

PivotDf (寬表格):
Data_ID  True_Label  llama3.2_1b_EMO01  llama3.2_1b_RAR01  ...
0         1           1                  1
1         0           0                  1
```

Pivot 後，`PromptCmbEval` 可以直接以欄名作為「模型/prompt 組合」的識別符，無需複雜的 join 操作即可計算指標。`fillna(-1)` 確保任何未被執行或解析失敗的組合以統一的哨兵值填補，不影響其他組合的評估。

---

### 3.7 評估層：`PromptCmbEval`

採用**兩類輸出分離**的設計：

- **定量報表**：`eval_summary.csv`，各模型 Accuracy / Precision / Recall / F1 / MCC，按 F1 降序排列
- **視覺化**：每個模型/prompt 組合的混淆矩陣 PNG + 全局對錯熱圖

**Upper Bound 分析**是一個較少見但很有價值的設計：識別出「所有模型/prompt 組合都答錯」的樣本（Hard Samples），計算出理論上限，讓實驗者判斷效能瓶頸是「模型/Prompt 策略不夠好」還是「資料集本身存在無法解決的困難樣本」。

---

## 四、SOLID 原則符合度評估

| 原則 | 評估 | 說明 |
|---|---|---|
| **S** 單一職責 | ✅ 良好 | 六個組件各司其職：TaskBuilder 不寫檔、OutputParser 不評估、Evaluate 不做 Pivot。每個類別的修改原因只有一個。 |
| **O** 開放封閉 | ⚠️ 部分符合 | Prompt 格式可透過 YAML `taskTemplate` 擴充，無需改程式碼。但若要新增評估指標，需直接修改 `doCalcPromptCmbMetrics`，缺少插件式擴充機制。 |
| **L** 里氏替換 | ✅ N/A | 目前無繼承關係，均為組合（Composition）設計，此原則不適用。 |
| **I** 介面隔離 | ✅ 良好 | `LLMEngine` 的使用者只需傳入 `taskDictList` 並呼叫 `doExecuteTaskBatches`，不需要理解其內部的 `OllamaClient`、Semaphore 或 Lock 機制。 |
| **D** 依賴反轉 | ⚠️ 部分符合 | `ExperimentPipeline` 直接實例化具體類別（如 `LLMEngine`、`OutputParser`），而非依賴抽象介面。當需要替換底層推論引擎（如從 Ollama 換為 vLLM）時，必須修改 `Pipeline.py`。 |

---

## 五、容錯機制（Resilience）

| 層次 | 機制 | 實作位置 |
|---|---|---|
| **配置層** | Pydantic 強型別驗證，錯誤在啟動時立即暴露 | `schemas.LLMAppConfig` |
| **任務層** | Checkpoint-Resume，任務以 append 模式逐筆持久化 | `Pipeline.doGetCompletedTasks` + `LLMEngine.doProcessSingleTask` |
| **網路層** | 指數退避自動重試（最多 3 次），持續失敗記錄 Error 字串不中斷流程 | `OllamaClient.doGenerate` (tenacity) |
| **解析層** | 解析失敗標記 -1 哨兵值，不中斷整體流程；空值保護（`pd.isna` 檢查） | `OutputParser.doExtractAnswers` |
| **資料層** | `doLoadData` / `doPivotData` 以 `raise PipelineError from e` 保留完整 traceback | `LLMResultProcessor` |
| **頂層** | `try/except` 捕捉所有 `Exception`，以 `logging.critical(..., exc_info=True)` 記錄完整 stack trace | `call_LLM.startLLMPipeline` |

---

## 六、效能設計

**主要效能瓶頸：** Ollama API 的推論延遲，這是 I/O-bound 問題，適合用非同步並發解決。

系統採用兩個維度的並發控制：

- `concurrencyPerModel: 8` — 單模型最多同時發出 8 個 API 請求
- `maxConcurrentModels: 1` — 同時最多只有 1 個模型在運行（避免 VRAM 競爭）

**批次設計（`pairNumbers: 10`）** 將 API 呼叫次數降低約一個數量級，是最直接的效能優化手段。

**潛在瓶頸：** `OutputParser.doParse` 和 `LLMResultProcessor.doPivotData` 目前是單執行緒的 pandas 操作，當資料量（論文數 × 模型數 × Prompt 數）大幅增長時，可能成為瓶頸。目前規模下（數千筆），這不是問題。

---

## 七、已知限制與技術風險

### 風險一：OutputParser 的脆弱性

Regex 解析對 LLM 輸出格式高度敏感。當 LLM 輸出格式不符預期時，`doExtractAnswers` 只能以 `-1` 標記失敗，目前無自動重送或二次解析機制。若整批回應格式不符，此 batch 的所有 pair 都會被標記為 `-1`，導致有效樣本數下降。

**建議方向：** 改用 Ollama 的 structured output（JSON mode）強制 LLM 回傳格式化回應，從根本上消除 Regex 解析的不確定性。

### 風險二：`Data_ID` 語義不一致

`TaskBuilder` 中，`batchData` 儲存的是 DataFrame 的原始索引（`orig_idx = idx`）作為 `Data_ID`，而非 CSV 中的 ID 欄位值。`OutputParser` 以此值寫入 `Data_ID` 欄位，但欄名暗示它是資料的業務 ID。這在 `testLimits` 啟用時不影響，但若資料集中 DataFrame 索引與業務 ID 不連續，可能造成後續 merge 操作（`LLMResultProcessor.doSaveData`）的對應錯誤。

### 風險三：推論後端耦合

`OllamaClient` 硬編碼對應 `/api/chat` 的 Ollama API 格式，若要切換至其他推論後端（如 vLLM、OpenAI API、HuggingFace TGI），需修改 `OllamaClient.doGenerate` 的 payload 結構。

**建議方向：** 將 `doGenerate` 抽象為 `BaseLLMClient` 介面，以支援多後端切換，同時也讓單元測試可以注入 Mock Client，不需要真實的 Ollama 服務。

---

## 八、總結

本系統在其定位範疇內（本地 LLM 評估實驗引擎）呈現出清晰的架構思維：

- 配置驗證前置，避免執行期的配置錯誤
- Checkpoint 機制讓長時間實驗具備容錯能力
- 雙層 Semaphore 在資源利用與系統穩定性之間取得平衡
- 各組件單一職責，維護成本低

主要的改進空間集中在：OutputParser 的解析策略（考慮結構化輸出）、`OllamaClient` 的後端抽象化，以及 `Data_ID` 語義的修正。
