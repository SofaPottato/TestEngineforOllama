# 專案架構與系統設計說明書
## LLM-Based Classification Evaluation System

**版本：** 2.0　｜　**作者：** Chen BOBO　｜　**日期：** 2026-04-14

---

## 一、系統定位與業務目標

本系統是一套針對**分類型 NLP 任務**（目前主用於生物醫學文本關係抽取，Chemical-Induced Disease）的 LLM 評估框架。其核心任務是：對多個本地部署的 LLM（透過 Ollama 提供服務），以多種 Prompt 策略進行組合測試，自動化量測各組合的精準度，並產出可供分析的統計報表與視覺化圖表。

**這不是一個推論服務，而是一個實驗評估引擎。** 設計目標是快速迭代 Prompt 策略、橫向比較模型表現、以及透過可重現的批量實驗取得可靠的量化結論。

相比 1.0 版本，本次設計引入了**資料集解耦的前處理階段**，使 Pipeline 本身不再綁定任何特定資料集（如 BC5CDR），可透過撰寫對應的前處理腳本快速接入新資料集。

---

## 二、整體架構概覽

```
┌───────────────────────────────────────────────────────┐
│        Stage 0: 資料集前處理（Dataset-Specific）      │
│                                                       │
│   Raw CSV (BC5CDR / Other)                            │
│        │                                              │
│        ▼                                              │
│   preprocess/bc5cdr.py（或對應資料集腳本）            │
│        │                                              │
│        ▼                                              │
│   tasks.csv                                           │
│   ─ taskID    （任務唯一識別碼）                      │
│   ─ context   （JSON dict — 共享欄位）                │
│   ─ items     （JSON array — 每筆帶 id/label + 自訂） │
└───────────────────────┬───────────────────────────────┘
                        │ 標準化資料合約
                        ▼
┌───────────────────────────────────────────────────────┐
│      Stage 1~6: 通用推論與評估 Pipeline               │
│                                                       │
│  call_LLM.py (CLI Entry)                              │
│       │                                               │
│       ▼                                               │
│  utils.ReadLLMConfig → Pydantic LLMAppConfig          │
│       │                                               │
│       ▼                                               │
│  Pipeline.ExperimentPipeline                          │
│   ├─ Step 1: doLoadTaskCsv / doLoadPromptCmb          │
│   ├─ Step 2: doBuildLLMTasks                          │
│   │          └─ PromptRenderer.render (template 填充) │
│   ├─ Step 3: LLMEngine.doExecuteTaskBatches (async)   │
│   ├─ Step 4: OutputParser.doParse (Regex)             │
│   ├─ Step 5: LLMResultProcessor.doCleanAndMerge       │
│   └─ Step 6: PromptCmbEval.doEval / doSaveResults     │
└───────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Pivot Wide CSV  Confusion PNG   eval_summary.csv
```

系統採用**「一階段前處理 + 六階段主 Pipeline」**的分層架構。前處理負責資料集專屬的欄位轉換，產出統一格式的 `tasks.csv`；主 Pipeline 只認識這個標準格式，不知道也不需要知道原始資料的結構。

---

## 三、Stage 0：資料集前處理邏輯

### 3.1 設計動機

1.0 版本將資料集欄位（`E1_Name`, `Relation_Type` 等）硬編碼在 `TaskBuilder` 內部，導致換資料集必須大改 Pipeline 程式碼。2.0 版本把這段邏輯抽出到獨立的前處理腳本，讓主 Pipeline 成為**資料集無關**（dataset-agnostic）的通用引擎。

### 3.2 標準 Task CSV 格式

所有前處理腳本的產物都必須符合此三欄規範：

| 欄位 | 型別 | 內容 |
|------|------|------|
| `taskID` | str | 任務唯一識別碼（斷點續傳用） |
| `context` | JSON dict | 本任務「共享」的欄位（對應 `taskTemplate` 的佔位符） |
| `items` | JSON array | 本任務包含的 item 清單；每筆必帶 `id` / `label`，其餘為自訂欄位（對應 `itemTemplate`） |

**範例（BC5CDR 批次模式，batchSize=2）：**

```csv
taskID,context,items
12345_0,"{""title"": ""Aspirin study"", ""abstract"": ""...""}","[{""id"":""1"",""label"":""cid"",""e1"":""Aspirin"",""e2"":""Headache""},{""id"":""2"",""label"":""0"",""e1"":""Aspirin"",""e2"":""Cancer""}]"
```

### 3.3 前處理腳本的兩種模式

以 [preprocess/bc5cdr.py](../preprocess/bc5cdr.py) 為參考實作：

#### 批次模式（`batchSize > 1`）
- 依 `PMID`（論文 ID）分組，每組內以 `batchSize` 切片
- `context` 僅放共享欄位（Title、Abstract）
- `items` 陣列存放每個 pair 的個別欄位（e1、e2、id、label）
- 優點：讓 LLM 在同一份 Abstract 上下文中回答多個問題，API 呼叫數降低約 `batchSize` 倍

```python
context = {'title': ..., 'abstract': ...}
items = [
    {'id': ..., 'label': ..., 'e1': ..., 'e2': ...},
    {'id': ..., 'label': ..., 'e1': ..., 'e2': ...},
]
```

#### 單筆模式（`batchSize == 1`）
- 每列原始資料獨立成一個 task
- `context` 合併所有欄位（包含 e1、e2）
- `items` 只保留 `{id, label}`，其餘欄位已在 context 內

```python
context = {'title': ..., 'abstract': ..., 'e1': ..., 'e2': ...}
items = [{'id': ..., 'label': ...}]
```

### 3.4 新資料集接入流程

要讓系統支援新資料集，開發者只需要：

1. 在 `preprocess/` 建立新腳本（可複製 `bc5cdr.py` 修改）
2. 調整 `requiredCols` 與欄位映射邏輯，輸出符合「taskID / context / items」三欄規範的 CSV
3. 對應更新 `configs/*.yaml` 的 `taskTemplate` / `itemTemplate`

**主 Pipeline（`llm_modules/` 下所有檔案）完全不需要改動。**

---

## 四、Config Template 與資料的對應關係

這是理解整個系統如何「組 prompt」的關鍵。

### 4.1 兩個 Template 的職責

| Template | 來源於 config | 填入的資料 | 使用時機 |
|----------|---------------|-----------|---------|
| `taskTemplate` | `llm_config.yaml` | `context` dict 的 key | 每個 task 套一次 |
| `itemTemplate` | `llm_config.yaml` | `items` 陣列中每個 item 的欄位 | 批次模式下每個 item 套一次 |

### 4.2 佔位符對應規則（BC5CDR 範例）

**config（llm_config.yaml）：**

```yaml
taskTemplate: |
  Title: {title}
  Abstract: {abstract}

  Task: Determine if the Chemical induces the Disease...

  Pairs to analyze:
  {items}                # ← 批次展開佔位符（特殊）

itemTemplate: "{i}: Chemical: {e1} | Disease: {e2}\n"
```

**tasks.csv 一列：**

```
context = {"title": "Aspirin study", "abstract": "..."}
items   = [
    {"id":"1", "label":"cid", "e1":"Aspirin", "e2":"Headache"},
    {"id":"2", "label":"0",   "e1":"Aspirin", "e2":"Cancer"}
]
```

**對應關係表：**

| Template 中 | 從哪裡取值 | 備註 |
|-------------|-----------|------|
| `{title}` | `context["title"]` | 任意自訂 key，與前處理腳本的 dict key 一致 |
| `{abstract}` | `context["abstract"]` | 同上 |
| `{items}` | 由 `itemTemplate` 渲染後的字串 | **保留字**，代表「批次展開位置」 |
| `{e1}` | `item["e1"]` | 每個 item 獨立渲染 |
| `{e2}` | `item["e2"]` | 同上 |
| `{i}` | `PromptRenderer` 自動產生 | **保留字**，從 1 開始的流水號 |

**`id` / `label` 不會傳入 template**——它們由系統內部使用（`id` 做樣本追蹤、`label` 做評估基準）。

### 4.3 渲染流程（由 `PromptRenderer.render` 執行）

```
┌───── 批次模式（itemTemplate 存在且 items > 1）─────┐
│                                                    │
│  1. 對每個 item 套 itemTemplate（排除 id/label）   │
│     → 串接成 itemsText                             │
│                                                    │
│  2. 以 context 的 key-value 填 taskTemplate        │
│                                                    │
│  3. 把 {items} 替換為 itemsText                    │
└────────────────────────────────────────────────────┘

┌───── 單筆模式（其餘情況）──────────────────────────┐
│                                                    │
│  1. 合併 context + items[0]（排除 id/label）       │
│                                                    │
│  2. 以合併後的 dict 填 taskTemplate                │
│     （itemTemplate 被忽略）                        │
└────────────────────────────────────────────────────┘
```

### 4.4 最終渲染結果（批次模式）

```
Title: Aspirin study
Abstract: ...

Task: Determine if the Chemical induces the Disease...

Pairs to analyze:
1: Chemical: Aspirin | Disease: Headache
2: Chemical: Aspirin | Disease: Cancer

IMPORTANT OUTPUT RULES:
...
```

### 4.5 設計意圖

- **前處理決定 key 的名字**，config template 用 `{}` 對應 → 兩者共同構成「資料與 prompt 的合約」
- 更換 prompt 格式 = 只改 YAML，不動程式
- 更換資料集欄位 = 只改前處理 + 對應更新 template，不動 Pipeline
- `PromptRenderer` 是獨立類別（[llm_modules/PromptRenderer.py](../llm_modules/PromptRenderer.py)），未來換模板引擎（如 Jinja2）只影響此一檔案

---

## 五、組件職責分析

### 5.1 配置層：`schemas.py` + `utils.ReadLLMConfig`

採用 Pydantic `BaseModel` 作為 schema 驗證引擎：

```
YAML File → yaml.safe_load() → dict → LLMAppConfig(**dict)
                                              ↓
                              PathsConfig (field_validator: 自動建立 Dir)
                              OllamaServerConfig
                              LabelMapConfig
                              LLMAppConfig (頂層聚合)
```

**設計優點：**

- `PathsConfig.ensureDirectories` 透過 `field_validator('*')` 在驗證階段自動建立所有帶 `Dir` 後綴的目錄，確保後續流程不會因「目錄不存在」失敗
- 必填欄位以 `Field(...)` 標記（如 `taskTemplate`），配置錯誤在啟動階段就暴露（Fail Fast），而非在執行時才崩潰

### 5.2 協調層：`Pipeline.ExperimentPipeline`

Pipeline 是整個系統的**指揮中心**，不包含業務邏輯，只負責協調各組件的執行順序與資料傳遞。包含**斷點續傳機制**：

```python
# Pipeline.doGetCompletedTasks()
completedIDs = set(checkpointDf['taskID'].dropna().astype(str).tolist())

# Pipeline.doBuildLLMTasks
fullTaskID = f"{model}::{promptID}::{taskBaseID}"
if fullTaskID in completedIDs:
    skipped += 1; continue
```

**機制：** `taskID` 格式為 `{model}::{promptID}::{taskBaseID}`，三個維度唯一標識一個任務。`LLMEngine` 以 append 模式逐筆寫入暫存 CSV，中斷重啟時僅未完成的任務會被重跑，這對數小時級別的實驗至關重要。

### 5.3 渲染層：`PromptRenderer`（新增）

獨立的 prompt 渲染類別，職責單純：把 `context` 與 `items` 填入 template 產出最終字串。

詳見 §4，此處不再重複。

### 5.4 非同步推論引擎：`OllamaEngine`

架構複雜度最高的組件，採用**雙層 Semaphore 限流設計**：

```
Level 1: modelConcurrencySemaphore（全域）
  └── 控制同時運行的模型數量 (maxConcurrentModels=1)
      → 防止多個大型模型同時佔用 VRAM 導致 OOM

Level 2: modelSemaphoreDict[model]（per-model）
  └── 控制單一模型的最大並發請求數 (concurrencyPerModel=8)
      → 防止對單一模型過度併發導致 Ollama 排隊崩潰
```

**`asyncio.as_completed` vs `asyncio.gather`：** 採用前者，讓先完成的任務優先更新 `tqdm` 進度條，對長時間推論的 UX 有顯著改善。

**重試機制：** `tenacity` 裝飾器對 `httpx.RequestError / HTTPStatusError / ReadTimeout` 進行最多 3 次、指數退避（1s → 2s → 4s）的自動重試，設 `reraise=True` 避免持續性故障被靜默吞掉。

**並發寫入保護：** `asyncio.Lock()` 保護 CSV 寫入，確保多個 coroutine 完成時資料不交錯。

### 5.5 輸出解析層：`OutputParser`

系統中**最脆弱的環節**，其穩健性完全依賴 LLM 輸出格式的一致性。

`doExtractAnswers` 的策略分模式處理：

- **單筆模式（batchSize==1）**：對整段回應做 `yes/positive/no/negative/none` 關鍵字掃描
- **批次模式**：以 regex 切分編號區塊，逐段掃描

```python
# 批次模式的切分 pattern（避免把 "No." 誤判為題號）
blocksList = re.split(r'\n\s*(?:\s+|No\.?\s*)?\d+\s*[:.)-]', text, flags=re.IGNORECASE)
```

解析失敗的結果標記為 `-1`（而非 `0`），確保「無法判斷」不會被誤算為「負類」。

輸出 CSV 的欄位：`dataID`, `Model`, `promptID`, `trueLabel`, `predLabel`, `rawOutput`, [自訂欄位...]。

### 5.6 資料處理層：`LLMResultProcessor`

**Long-to-Wide Pivot** 是核心操作：

```
InputDf (長表格):
Model         promptID   dataID   predLabel
llama3.2_1b   EMO01      0        1
llama3.2_1b   EMO01      1        0
llama3.2_1b   RAR01      0        1
...

↓ pivot_table(index=[dataID, trueLabel, ...], columns='Feature_Name', values='predLabel')

PivotDf (寬表格):
dataID  trueLabel  llama3.2_1b_EMO01  llama3.2_1b_RAR01  ...
0        1          1                  1
1        0          0                  1
```

`Feature_Name` 由 `Model + "_" + promptID` 組合而成，Pivot 後每欄代表一個「模型/Prompt 組合」。`fillna(-1)` 確保未執行或解析失敗的組合以哨兵值填補。

`trueLabel` 經 `labelMap` 標準化為 `0/1/-1`（config 的 `labelMap.positive` / `labelMap.negative` 控制映射規則）。

### 5.7 評估層：`PromptCmbEval`

採用**兩類輸出分離**設計：

- **定量報表**：`eval_summary.csv`，各組合的 Accuracy / Precision / Recall / F1 / MCC，按 `f1Score` 降序排列
- **視覺化**：每個組合的混淆矩陣 PNG + 全局對錯熱圖

**Upper Bound 分析**：識別出「所有組合都答錯」的樣本（Hard Samples），計算理論上限，讓實驗者判斷瓶頸來自「模型/Prompt 策略不夠好」還是「資料集本身有無解樣本」。

---

## 六、SOLID 原則符合度評估

| 原則 | 評估 | 說明 |
|---|---|---|
| **S** 單一職責 | ✅ 良好 | 組件各司其職：前處理、PromptRenderer、Engine、Parser、Processor、Evaluator 職責邊界清晰。2.0 版將 prompt 渲染從 Pipeline 抽出為獨立類別，進一步強化此原則。 |
| **O** 開放封閉 | ✅ 改善 | 接入新資料集只需新增前處理腳本 + 改 config，無需修改 Pipeline。但若要新增評估指標仍需改 `doCalcPromptCmbMetrics`。 |
| **L** 里氏替換 | ✅ N/A | 目前無繼承關係，均為組合設計。 |
| **I** 介面隔離 | ✅ 良好 | `LLMEngine` 的使用者只需傳入 `taskDictList`，不需理解其內部的 Semaphore/Lock 機制。 |
| **D** 依賴反轉 | ⚠️ 部分符合 | `ExperimentPipeline` 直接實例化具體類別（如 `LLMEngine`），未依賴抽象介面。替換推論後端（如 vLLM）需修改 Pipeline。 |

---

## 七、容錯機制（Resilience）

| 層次 | 機制 | 實作位置 |
|---|---|---|
| **配置層** | Pydantic 強型別驗證，錯誤在啟動時立即暴露 | `schemas.LLMAppConfig` |
| **任務層** | Checkpoint-Resume，任務以 append 模式逐筆持久化 | `Pipeline.doGetCompletedTasks` + `OllamaEngine` |
| **網路層** | 指數退避自動重試（最多 3 次） | `OllamaClient.doGenerate` (tenacity) |
| **解析層** | 解析失敗標記 `-1` 哨兵值，不中斷整體流程 | `OutputParser.doExtractAnswers` |
| **資料層** | `doLoadData` / `doPivotData` 以 `raise PipelineError from e` 保留完整 traceback | `LLMResultProcessor` |
| **頂層** | `try/except` 捕捉所有 `Exception`，`logging.critical(..., exc_info=True)` 記錄完整 stack trace | `call_LLM.startLLMPipeline` |

---

## 八、效能設計

**主要瓶頸：** Ollama API 的推論延遲，屬於 I/O-bound 問題，適合非同步並發解決。

系統採用兩個維度的並發控制：

- `concurrencyPerModel: 8` — 單模型最多同時發出 8 個 API 請求
- `maxConcurrentModels: 1` — 同時最多只有 1 個模型在運行（避免 VRAM 競爭）

**批次設計（`batchSize` 由前處理決定）** 將 API 呼叫次數降低一個數量級，是最直接的效能優化手段。

---

## 九、已知限制與技術風險

### 風險一：OutputParser 的脆弱性
Regex 解析對 LLM 輸出格式高度敏感。若整批回應格式不符，此 batch 的所有 item 都會被標記為 `-1`，有效樣本數下降。

**建議方向：** 改用 Ollama 的 structured output（JSON mode）強制 LLM 回傳格式化回應。

### 風險二：解析策略綁定二元分類
`OutputParser`（yes/no 關鍵字）、`LLMResultProcessor`（predLabel ∈ {-1,0,1}）、`Evaluate`（predValueSet = {-1,0,1}）三處都假設二元分類任務。支援多類別或抽取任務需引入「任務類型策略」機制。

### 風險三：推論後端耦合
`OllamaClient` 硬編碼 Ollama `/api/chat` 的 payload 格式。切換至 vLLM / OpenAI API / HuggingFace TGI 需修改 `doGenerate`。

**建議方向：** 抽象 `BaseLLMClient` 介面，支援多後端切換，同時讓單元測試可以注入 Mock Client。

---

## 十、總結

2.0 版本的核心架構進化是**資料集解耦**：

- **前處理階段獨立**：`preprocess/<dataset>.py` 承擔所有資料集專屬邏輯，產出標準格式 `tasks.csv`
- **PromptRenderer 抽離**：prompt 渲染不再是 Pipeline 的內部方法，成為可獨立測試、可替換的類別
- **Template ↔ 資料合約明確化**：config 的 `{key}` 與前處理輸出的 `context/items` 欄位一對一對應，職責邊界清晰
- **欄位命名統一 camelCase**：CSV 欄位（`dataID`, `trueLabel`, `predLabel` 等）與 Python 變數命名一致，降低跨層閱讀成本

剩餘的改進空間集中在：`OutputParser` 的結構化輸出、二元分類假設的解耦，以及推論後端的抽象化。
