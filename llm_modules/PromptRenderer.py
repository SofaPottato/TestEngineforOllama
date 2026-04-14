from typing import Dict, List, Optional


class PromptRenderer:
    def __init__(self, taskTemplate: str, itemTemplate: Optional[str] = None):
        """
        負責將 context 與 items 填入 template，產出最終的 userPrompt 字串。

        :param taskTemplate: 任務層級的文字模板；{key} 對應 context 欄位，{items} 對應批次展開
        :param itemTemplate: 單筆 item 的格式化模板（如 '{i}: {e1} | {e2}\n'）；None 代表單筆模式
        """
        self.taskTemplate = taskTemplate
        self.itemTemplate = itemTemplate

    def render(self, context: Dict, items: List[Dict]) -> str:
        """
        將 context 與 items 渲染為最終的 userPrompt。

        批次模式（itemTemplate 存在且 items > 1）：
          每個 item 透過 itemTemplate 渲染後串接，填入 {items} 佔位符。

        單筆模式：
          context 與 items[0] 的欄位合併後，直接填入 taskTemplate。

        :param context: Task CSV 的 context JSON dict
        :param items: Task CSV 的 items JSON array
        :return: 渲染完成的 userPrompt 字串
        """
        if self.itemTemplate and len(items) > 1:
            return self._renderBatch(context, items)
        return self._renderSingle(context, items)

    def _renderBatch(self, context: Dict, items: List[Dict]) -> str:
        itemsText = ""
        for i, item in enumerate(items, 1):
            renderFields = {k: v for k, v in item.items() if k not in ('id', 'label')}
            itemsText += self.itemTemplate.format(i=i, **renderFields)

        prompt = self.taskTemplate
        for k, v in context.items():
            prompt = prompt.replace('{' + k + '}', str(v))
        prompt = prompt.replace('{items}', itemsText)
        return prompt

    def _renderSingle(self, context: Dict, items: List[Dict]) -> str:
        allFields = dict(context)
        if items:
            allFields.update({k: v for k, v in items[0].items() if k not in ('id', 'label')})

        prompt = self.taskTemplate
        for k, v in allFields.items():
            prompt = prompt.replace('{' + k + '}', str(v))
        return prompt
