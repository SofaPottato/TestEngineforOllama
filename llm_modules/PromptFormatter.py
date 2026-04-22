import logging
from typing import Dict, List, Optional


class _SafeDict(dict):
    """
    給 str.format_map 用的容錯字典：
    - 缺漏的 key 回傳原始 '{key}' 佔位符（保留可見性，方便人工發現缺值）
    - 值會強制轉成 str，避免 None / 數值型導致格式化異常
    """
    def __missing__(self, key):
        logging.warning(f"PromptFormatter: 模板佔位符 {{{key}}} 在資料中不存在，保留原樣。")
        return '{' + key + '}'


def _safeFormat(template: str, fields: Dict) -> str:
    """
    用 format_map 套版，並把所有值轉 str。
    比 str.replace 安全：
    - 不會被資料內容裡的 '{xxx}' 字串污染
    - 缺欄位時會記 warning，不會靜默吞掉
    """
    safeFields = _SafeDict({k: ('' if v is None else str(v)) for k, v in fields.items()})
    return template.format_map(safeFields)


class PromptFormatter:
    def __init__(self, taskTemplate: str, pairTemplate: Optional[str] = None,
                 pairColumns: Optional[List[str]] = None):
        """
        負責將 context 與 pairs 填入 template，產出最終的 userPrompt 字串。

        :param taskTemplate: 任務層級的文字模板；{key} 對應 context 欄位，{pairs} 對應批次展開
        :param pairTemplate: 單筆 pair 的格式化模板（如 '{i}: {e1} | {e2}\n'）；None 代表單筆模式
        :param pairColumns: 從 pair dict 中取出的欄位名稱清單；None 時取除 id/label 以外的全部欄位
        """
        self.taskTemplate = taskTemplate
        self.pairTemplate = pairTemplate
        self.pairColumns = pairColumns

    def _extractPairFields(self, pair: Dict) -> Dict:
        if self.pairColumns:
            return {k: pair[k] for k in self.pairColumns if k in pair}
        return {k: v for k, v in pair.items() if k not in ('id', 'label')}

    def format(self, context: Dict, pairs: List[Dict]) -> str:
        """
        將 context 與 pairs 格式化為最終的 userPrompt。

        批次模式（pairTemplate 存在且 pairs > 1）：
          每個 pair 透過 pairTemplate 格式化後串接，填入 {pairs} 佔位符。

        單筆模式：
          context 與 pairs[0] 的欄位合併後，直接填入 taskTemplate。

        :param context: Task CSV 的 context dict
        :param pairs: Task CSV 的 pairs JSON array
        :return: 格式化完成的 userPrompt 字串
        """
        if self.pairTemplate and len(pairs) > 1:
            return self._formatBatch(context, pairs)
        return self._formatSingle(context, pairs)

    def _formatBatch(self, context: Dict, pairs: List[Dict]) -> str:
        pairsText = ""
        for i, pair in enumerate(pairs, 1):
            pairsText += _safeFormat(self.pairTemplate, {'i': i, **self._extractPairFields(pair)})

        # context 與 pairs 兩段分開填，避免資料中含 '{pairs}' 字面量被誤展開
        prompt = _safeFormat(self.taskTemplate, {**context, 'pairs': '{pairs}'})
        prompt = prompt.replace('{pairs}', pairsText)
        return prompt

    def _formatSingle(self, context: Dict, pairs: List[Dict]) -> str:
        allFields = dict(context)
        if pairs:
            allFields.update(self._extractPairFields(pairs[0]))
        return _safeFormat(self.taskTemplate, allFields)
