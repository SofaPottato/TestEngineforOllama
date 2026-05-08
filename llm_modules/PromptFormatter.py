"""
Prompt 模板渲染。用 str.format_map，缺欄位直接拋 KeyError 停止執行。
"""
from typing import Dict, List, Optional
from .schemas import RESERVED_PAIR_FIELDS


def _safeFormat(template: str, fields: Dict) -> str:
    try:
        return template.format_map({fieldName: ('' if fieldValue is None else str(fieldValue)) for fieldName, fieldValue in fields.items()})
    except KeyError as e:
        raise KeyError(f"[Formatter] 模板佔位符 {e} 在資料中不存在") from e


class PromptFormatter:
    """
    將 context 與 pairs 填入 taskTemplate / pairTemplate，產出 userPrompt。
    pairTemplate 存在 → 批次模式（多 pair 展開後塞入 {pairs}）；否則 → 單筆模式（context + pairs[0] 合併填入）。
    """
    def __init__(self, taskTemplate: str, pairTemplate: Optional[str] = None,
                 pairColumns: Optional[List[str]] = None):
        self.taskTemplate = taskTemplate
        self.pairTemplate = pairTemplate
        self.pairColumns = pairColumns

    def format(self, contextDict: Dict, pairs: List[Dict]) -> str:
        """pairTemplate 存在 → 批次模式；否則 → 單筆模式。"""
        if self.pairTemplate:
            return self._formatBatch(contextDict, pairs)
        return self._formatSingle(contextDict, pairs)

    def _formatBatch(self, contextDict: Dict, pairs: List[Dict]) -> str:
        """批次模式：每個 pair 渲染後拼接，整段填入 {pairs} 佔位符。"""
        pairsText = ""
        for i, pairDict in enumerate(pairs, 1):
            pairsText += _safeFormat(self.pairTemplate, {'i': i, **self._extractPairFields(pairDict)})
        return _safeFormat(self.taskTemplate, {**contextDict, 'pairs': pairsText})

    def _formatSingle(self, contextDict: Dict, pairs: List[Dict]) -> str:
        """單筆模式：context 與 pairs[0] 合併後直接填入 taskTemplate。pair 欄優先，同名時覆蓋 context。"""
        allFieldDict = dict(contextDict)
        if pairs:
            allFieldDict.update(self._extractPairFields(pairs[0]))
        return _safeFormat(self.taskTemplate, allFieldDict)

    def _extractPairFields(self, pairDict: Dict) -> Dict:
        """從 pair dict 抽取要送進模板的欄位，排除 RESERVED_PAIR_FIELDS 中的內部欄位。"""
        if self.pairColumns:
            return {colName: pairDict[colName] for colName in self.pairColumns if colName in pairDict}
        return {fieldName: fieldValue for fieldName, fieldValue in pairDict.items() if fieldName not in RESERVED_PAIR_FIELDS}
