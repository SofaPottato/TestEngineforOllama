# Ollama_testengine
An engine for ollama 
#LLM部分斷續
#ML部分路徑自訂boruta
#prompt 組合(命名camel格式)(輸出list與總結範例)(main_prompt generatodo )dictionarylist改為Dict與List路徑加Path 
#與LLM分開(callLLM API->輸出->)->main_feature->main_ML
#TODO:用戶自訂路徑，放所有東西 1.每個PROMPT 輸出一個檔案2.把輸出/結果併成DATA FRAME 路徑可自訂 3.EVAL清單 4.sampledata

工作日志3/13:
分離Prompt_generate 部分為main_prompt_generate
輸出prompt list
可自訂輸出路徑與檔案命名
新增:
prompt_config
main_prompt_generate
修改:
Prompt_manager
