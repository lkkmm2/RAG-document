"""中文文档标题识别。

清洗和切分都要判断"这一行是不是标题"，所以抽出来一处，避免两套正则漂移。
只做规则匹配，不上模型：制度类文档的编号体系非常稳定，
正则的准确率已经够用，而且可解释、可测试、零延迟。
"""

from __future__ import annotations

import re

# 层级越小越靠上：章 > 节/一、 > （一）/ 1.1 > 1.1.1
_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^第[一二三四五六七八九十百零〇\d]+[章篇部]"), 1),
    (re.compile(r"^第[一二三四五六七八九十百零〇\d]+节"), 2),
    (re.compile(r"^[一二三四五六七八九十百]+[、.．]"), 2),
    (re.compile(r"^[（(][一二三四五六七八九十百]+[）)]"), 3),
    (re.compile(r"^\d{1,2}\.\d{1,2}\.\d{1,2}[\s、.．]?\S"), 4),
    (re.compile(r"^\d{1,2}\.\d{1,2}[\s、.．]?\s*\S"), 3),
    (re.compile(r"^\d{1,2}[、.．]\s*\S"), 2),
]

MAX_HEADING_CHARS = 40


def match_heading(line: str) -> tuple[int, str] | None:
    """返回 (层级, 标题文本)；不是标题返回 None。

    标题的判定除了前缀，还要看长度和是否成句——
    "1. 学生应当在每学期开学后两周内完成注册。" 是正文，
    "1. 注册与学籍" 才是标题。
    """
    text = line.strip()
    if not text or len(text) > MAX_HEADING_CHARS:
        return None
    if text.endswith(("。", "！", "？", "；", "，", ".", "!", "?", ";")):
        return None
    if "。" in text:
        return None
    for pattern, level in _PATTERNS:
        if pattern.match(text):
            return level, text
    return None


HEADING_START_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百零〇\d]+[章节篇部分]"
    r"|[一二三四五六七八九十百]+[、.．]"
    r"|[（(][一二三四五六七八九十百]+[）)]"
    r"|\d{1,2}(?:\.\d{1,2}){0,2}[\s、.．]\s*\S)"
)
