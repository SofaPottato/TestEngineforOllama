# 專案架構與系統設計說明書
## LLM-Based Classification Evaluation System

**版本：** 2.1　｜　**作者：** Chen BOBO　｜　**日期：** 2026-04-17

---

## 一、系統定位與業務目標

本系統是一套針對**分類型 NLP 任務**（目前主用於生物醫學文本關係抽取，Chemical-Induced Disease）的 LLM 評估框架。核心任務是：對多個本地部署的 LLM（透過 Ollama 提供服務），以多種 Prompt 策略進行組合測試，自動化量測各組合的精準度，並產出可供分析的統計報表與視覺化圖表。

**這不是一個推論服務，而是一個實驗評估引擎。** 設計目標是快速迭代 Prompt 策略、橫向比較模型表現、以及透過可重現的批量實驗取得可靠的量化結論。

系統採用**「一階段前處理 + 六階段主 Pipeline」**的分層架構：前處理負責資料集專屬的欄位轉換，產出統一格式的 `tasks.csv`；主 Pipeline 只認識這個標準格式，不知道也不需要知道原始資料的結構。

---

## 二、整體架構概覽

```
┌─────────────────────────────────────────────────────┐
│      Stage 0: 資料集前處理（Dataset-Specific）      │
│                                                     │
│   Raw CSV (BC5CDR / Other)                          │
│        │                                            │
│        ▼                                            │
│   preprocess/bc5cdr.py（或對應資料集腳本）           │
│        │                                            │
│        ▼                                            │
│   tasks.csv                                         │
│   ─ taskID    （任務唯一識別碼，格式為 PMID）        │
│   ─ title     （文章標題）                          │
│   ─ abstract  （文章摘要）                          │
│   ─ pairs     （JSON array — 每筆帶 id/label/e1/e2）│
└────────────────────────┬────────────────────────────┘
                         │ 標準化資料合約
                         ▼
┌─────────────────────────────────────────────────────┐
│      Stage 1~6: 通用推論與評估 Pipeline             │
│                                                     │
│  call_LLM.py (CLI Entry)                            │
│       │                                             │
│       ▼                                             │
│  utils.ReadLLMConfig → Pydantic LLMAppConfig        │
│       │                                             │
│       ▼                                             │
│  Pipeline.ExperimentPipeline                        │
│   ├─ Step 1: doLoadTaskCsv / doLoadPromptCmb        │
│   │          └─ doSavePromptPreview（prompt 預覽）  │
│   ├─ Step 2: doBuildLLMTasks                        │
│   │          └─ _buildTaskBatches                   │
│   │             └─ PromptFormatter.format           │
│   ├─ Step 3: LLMEngine.doExecuteTaskBatches (async) │
│   ├─ Step 4: OutputParser.run (Regex)               │
│   ├─ Step 5: LLMResultProcessor.run (Pivot)         │
│   └─ Step 6: PromptCmbEval.run (Eval + Report)      │
└─────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    Pivot Wide CSV  Confusion PNG   eval_summary.csv
```

---

## 三、Stage 0：資料集前處理

### 3.1 設計動機

原版本將資料集欄位（`E1_Name`, `Relation_Type` 等）硬編碼在 Pipeline 內部，導致換資料集必須大改主程式碼。現版本把這段邏輯抽出到獨立的前處理腳本，讓主 Pipeline 成為**資料集無關**（dataset-agnostic）的通用引擎。

### 3.2 標準 Task CSV 格式

所有前處理腳本的產物都必須符合此規範：

| 欄位 | 型別 | 內容 |
|------|------|------|
| `taskID` | str | 任務唯一識別碼（斷點續傳用），BC5CDR 以 PMID 作為 taskID |
| `title` | str | 文章標題（對應 `taskTemplate` 的 `{title}` 佔位符） |
| `abstract` | str | 文章摘要（對應 `taskTemplate` 的 `{abstract}` 佔位符） |
| `pairs` | JSON array | 本任務包含的所有 entity pair；每筆必帶 `id` / `label`，其餘為自訂欄位（對應 `pairTemplate`） |

`contextColumns` 在 config 中宣告對應 context 的欄位名稱（預設 `["title", "abstract"]`），Pipeline 以此清單驗證 CSV 欄位並組成 context dict。

**pairs 中 `id` 的來源：**
BC5CDR 資料集本身沒有官方 pair ID，`id` 對應原始 `BCVCDR_Processed.csv` 的行號（row index），作為追蹤每筆預測結果的唯一識別碼。

**範例（pairNumber=2，某 PMID 有 3 個 pair，切成 2 批）：**

```csv
taskID,title,abstract,pairs
12345,Aspirin study,...,"[{""id"":""0"",""label"":""cid"",""e1"":""Aspirin"",""e2"":""Headache""},{""id"":""1"",""label"":""0"",""e1"":""Aspirin"",""e2"":""Cancer""}]"
```

### 3.3 前處理腳本（bc5cdr.py）

- 路徑寫死在腳本頂部（`INPUT_PATH` / `OUTPUT_PATH`），直接執行即可
- 依 PMID 分組（`sort=False` 保持原始順序），每組所有 pairs 整批輸出為一筆 task
- **切片（pairNumber）不在前處理做**，由 Pipeline 的 `_buildTaskBatches` 負責

```python
for pmid, group in df.groupby('PMID', sort=False):
    pairs = [{'id': ..., 'label': ..., 'e1': ..., 'e2': ...} for _, row in group.iterrows()]
    tasks.append({'taskID': str(pmid), 'title': ..., 'abstract': ..., 'pairs': json.dumps(pairs)})
```

### 3.4 新資料集接入

1. 在 `preprocess/` 建立新腳本，輸出符合上述欄位規範的 CSV
2. 更新 `configs/*.yaml` 的 `contextColumns`、`taskTemplate`、`pairTemplate`

**主 Pipeline（`llm_modules/`）完全不需要改動。**

---

## 四、Config 與資料的對應關係

### 4.1 三個 Template 相關設定

| 設定 | 來源 | 說明 |
|------|------|------|
| `contextColumns` | `llm_config.yaml` | Task CSV 中作為 context 的欄位名稱，對應 `taskTemplate` 的佔位符 |
| `taskTemplate` | `llm_config.yaml` | 任務層級模板；`{key}` 對應 context 欄位，`{pairs}` 為批次展開佔位符 |
| `pairTemplate` | `llm_config.yaml` | 單筆 pair 的格式化模板；`{i}` 為流水號，其餘對應 pairs 中各欄位 |

### 4.2 佔位符對應規則（BC5CDR 範例）

**llm_config.yaml：**

```yaml
contextColumns: ["title", "abstract"]

taskTemplate: |
  Title: {title}
  Abstract: {abstract}

  Task: Determine if the Chemical induces the Disease...

  Pairs to analyze:
  {pairs}

pairTemplate: |
  {i}: Chemical: {e1} | Disease: {e2}
```

**對應關係表：**

| 佔位符 | 資料來源 | 備註 |
|--------|---------|------|
| `{title}` | `row['title']` | contextColumns 宣告的欄位 |
| `{abstract}` | `row['abstract']` | contextColumns 宣告的欄位 |
| `{pairs}` | 由 `pairTemplate` 格式化後串接的字串 | **保留字**，代表批次展開位置 |
| `{e1}` | `pair["e1"]` | 每個 pair 獨立格式化 |
| `{e2}` | `pair["e2"]` | 每個 pair 獨立格式化 |
| `{i}` | `PromptFormatter` 自動產生 | **保留字**，從 1 開始的流水號 |

`id` / `label` 不傳入 template，由系統內部使用（`id` 做樣本追蹤、`label` 做評估基準）。

### 4.3 格式化流程（`PromptFormatter.format`）

```
┌───── 批次模式（pairTemplate 存在且 pairs > 1）─────┐
│                                                    │
│  1. 對每個 pair 套 pairTemplate（排除 id/label）   │
│     → 串接成 pairsText                             │
│                                                    │
│  2. 以 context dict 填 taskTemplate                │
│                                                    │
│  3. 把 {pairs} 替換為 pairsText                    │
└────────────────────────────────────────────────────┘

┌───── 單筆模式（pairNumber==1 或 pairTemplate 未設）┐
│                                                    │
│  1. 合併 context + pairs[0]（排除 id/label）       │
│                                                    │
│  2. 以合併後的 dict 填 taskTemplate                │
└────────────────────────────────────────────────────┘
```

### 4.4 最終格式化結果（批次模式範例）

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

---

## 五、組件職責分析

### 5.1 配置層：`schemas.py` + `utils.ReadLLMConfig`

採用 Pydantic `BaseModel` 作為 schema 驗證引擎：

```
YAML File → yaml.safe_load() → dict → LLMAppConfig(**dict)
                                              ↓
                              PathsConfig（model_validator：自動建立所有輸出目錄）
                              OllamaServerConfig
                              LabelMapConfig
                              LLMAppConfig（頂層聚合）
```

- 必填欄位以 `Field(...)` 標記，配置錯誤在啟動階段就暴露（Fail Fast）
- 輸出路徑未填時自動衍生到 `outputRoot` 底下（見 `PathsConfig._DEFAULT_NAMES`）
- `PathsConfig.resolveAndEnsureDirectories` 統一在驗證後建立所有需要的目錄

### 5.2 協調層：`Pipeline.ExperimentPipeline`

Pipeline 是整個系統的**指揮中心**，不包含業務邏輯，只負責協調各組件的執行順序與資料傳遞。

**Step 1 新增 Prompt 預覽輸出（`doSavePromptPreview`）：**
在推論開始前，將所有 `promptID × task` 組合格式化後的完整 prompt 存成 `prompt_preview.csv`，欄位為 `taskID / promptID / sysPrompt / userPrompt`，供人工確認 prompt 正確性。

**斷點續傳機制：**

```
taskID 格式：{model}::{promptID}::{taskBaseID}

doGetCompletedTasks() → 讀取 raw.csv → 取出所有已完成的 taskID → Set
doBuildLLMTasks()     → 對每個組合產生 fullTaskID → 在 Set 中 → 跳過
```

三個維度（model / promptID / taskBaseID）唯一標識一個任務。`LLMEngine` 以 append 模式逐筆寫入，中斷重啟時僅未完成的任務會被重跑。

### 5.3 切片與格式化層：`_buildTaskBatches` + `PromptFormatter`

`_buildTaskBatches` 負責將 Task CSV 的每一列按 `pairNumber` 切片，並呼叫 `PromptFormatter.format` 格式化每個 batch 的 userPrompt。回傳 `[(batchID, batchPairs, userPrompt), ...]`。

- `pairNumber == 1` 時：`taskID` 保持原始 PMID
- `pairNumber > 1` 時：`taskID` 衍生為 `{pmid}_{offset}`

`PromptFormatter` 是獨立類別（`llm_modules/PromptFormatter.py`），職責單純：把 context dict 與 pairs list 填入 template 產出字串。未來換模板引擎只影響此一檔案。

### 5.4 非同步推論引擎：`OllamaEngine`

架構複雜度最高的組件，採用**雙層 Semaphore 限流設計**：

```
Level 1: modelConcurrencySemaphore（全域）
  └── 控制同時運行的模型數量（maxConcurrentModels）
      → 防止多個大型模型同時佔用 VRAM 導致 OOM

Level 2: modelSemaphoreDict[model]（per-model）
  └── 控制單一模型的最大並發請求數（concurrencyPerModel）
      → 防止對單一模型過度併發
```

- `asyncio.as_completed`：先完成的任務優先更新進度條，UX 較 `asyncio.gather` 好
- `tenacity`：對 `httpx` 網路異常最多重試 3 次，指數退避（1s → 2s → 4s）
- `asyncio.Lock()`：保護 CSV 寫入，防止多 coroutine 並發時資料交錯

### 5.5 輸出解析層：`OutputParser`

系統中**最脆弱的環節**，穩健性完全依賴 LLM 輸出格式的一致性。

`doExtractAnswers` 的策略：
- **單筆模式**：對整段回應做 `yes/no` 關鍵字掃描
- **批次模式**：以 regex 切分編號區塊，逐段掃描

解析失敗標記為 `-1`（哨兵值），確保「無法判斷」不被誤算為「負類」。

輸出 CSV 欄位：`dataID`, `Model`, `promptID`, `trueLabel`, `predLabel`, `rawOutput`, [自訂欄位...]。

### 5.6 資料處理層：`LLMResultProcessor`

核心操作是 **Long-to-Wide Pivot**：

```
長表格（InputDf）:
Model         promptID   dataID   predLabel
llama3.2_1b   EMO01      0        1
llama3.2_1b   RAR01      0        1

↓ pivot_table(index=[dataID, trueLabel], columns='Feature_Name', values='predLabel')

寬表格（PivotDf）:
dataID  trueLabel  llama3.2_1b__EMO01  llama3.2_1b__RAR01
0        1          1                   1
```

`Feature_Name = Model + "__" + promptID`，每欄代表一個「模型/Prompt 組合」。`fillna(-1)` 確保未執行或解析失敗的組合以哨兵值填補。`trueLabel` 由 `labelMap` 標準化為 `0/1`。

### 5.7 評估層：`PromptCmbEval`

- **定量報表**：`eval_summary.csv`，各組合的 Accuracy / Precision / Recall / F1 / MCC，按 F1 降序排列
- **視覺化**：每個組合的混淆矩陣 PNG + 全局對錯熱圖
- **Upper Bound 分析**：識別「所有組合都答錯」的 Hard Samples，計算理論上限

---

## 六、容錯機制

| 層次 | 機制 | 實作位置 |
|------|------|---------|
| 配置層 | Pydantic 強型別驗證，錯誤在啟動時立即暴露 | `schemas.LLMAppConfig` |
| 任務層 | Checkpoint-Resume，以 append 模式逐筆持久化 | `Pipeline.doGetCompletedTasks` + `OllamaEngine` |
| 網路層 | 指數退避自動重試（最多 3 次） | `OllamaClient.doGenerate` (tenacity) |
| 解析層 | 解析失敗標記 `-1` 哨兵值，不中斷整體流程 | `OutputParser.doExtractAnswers` |
| 頂層 | `logging.critical(..., exc_info=True)` 記錄完整 stack trace | `call_LLM.startLLMPipeline` |

---

## 七、效能設計

**主要瓶頸：** Ollama API 推論延遲（I/O-bound），以非同步並發解決。

- `concurrencyPerModel`：單模型最多同時發出 N 個 API 請求（config 預設 2）
- `maxConcurrentModels`：同時最多只有 N 個模型在運行（config 預設 1，避免 VRAM 競爭）
- `pairNumber`：每個 task 包含多個 pair，降低 API 呼叫次數，是最直接的效能優化手段

---

## 八、已知限制與技術風險

### 風險一：OutputParser 的脆弱性
Regex 解析對 LLM 輸出格式高度敏感。整批格式不符時，該 batch 所有 pair 都會被標記為 `-1`。

**建議方向：** 改用 Ollama structured output（JSON mode）強制 LLM 回傳格式化回應。

### 風險二：解析策略綁定二元分類
`OutputParser`（yes/no 關鍵字）、`LLMResultProcessor`（predLabel ∈ {-1,0,1}）、`Evaluate` 三處都假設二元分類任務。支援多類別或抽取任務需引入「任務類型策略」機制。

### 風險三：推論後端耦合
`OllamaClient` 硬編碼 Ollama `/api/chat` 的 payload 格式。切換至 vLLM / OpenAI API 需修改 `doGenerate`。

**建議方向：** 抽象 `BaseLLMClient` 介面，支援多後端切換，並讓單元測試可以注入 Mock Client。
