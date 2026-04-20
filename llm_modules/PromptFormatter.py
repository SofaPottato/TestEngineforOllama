from typing import Dict, List, Optional


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
            pairsText += self.pairTemplate.format(i=i, **self._extractPairFields(pair))

        prompt = self.taskTemplate
        for k, v in context.items():
            prompt = prompt.replace('{' + k + '}', str(v))
        prompt = prompt.replace('{pairs}', pairsText)
        return prompt

    def _formatSingle(self, context: Dict, pairs: List[Dict]) -> str:
        allFields = dict(context)
        if pairs:
            allFields.update(self._extractPairFields(pairs[0]))

        prompt = self.taskTemplate
        for k, v in allFields.items():
            prompt = prompt.replace('{' + k + '}', str(v))
        return prompt
